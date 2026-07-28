"""内层 ReAct + Function Calling 循环。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from memopilot.extensions.events import EventBus
from memopilot.extensions.hooks import ToolExecutionRequest
from memopilot.runtime.contracts import ChatMessage, FunctionCall, ModelResponse, StreamDelta
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tool_search import ToolDiscoveryState, ToolSearchTool
from memopilot.runtime.tools import ToolObservation, ToolRegistry

_SUMMARY_PROMPT = """当前任务需要先暂停继续调用工具，请直接输出给用户看的中文阶段性回复。
必须基于已有上下文，不要编造结果。
必须包含四点：
1) 已经使用了哪些工具或操作，以及拿到了什么关键信息；
2) 当前已经做到哪一步；
3) 还缺什么信息或步骤；
4) 如果继续，下一步会怎么做。
可以提到工具名称和关键结果，但不要暴露 tool_call_id、schema、内部 prompt 或原始参数 JSON。
禁止输出"已达到最大迭代次数"这类模板句；不要输出 JSON。"""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallRecord:
    iteration: int
    call: FunctionCall
    observation: ToolObservation


@dataclass(frozen=True)
class ReActResult:
    reply: str
    messages: tuple[ChatMessage, ...]
    iterations: int
    tool_chain: tuple[ToolCallRecord, ...]
    exit_reason: str
    infrastructure_error: str | None = None
    thinking: str | None = None
    prompt_tokens: int = 0
    prompt_cache_hit_tokens: int = 0


@dataclass(frozen=True, slots=True)
class BeforeStepControl:
    extra_hints: tuple[str, ...] = ()
    early_stop: bool = False
    early_stop_reply: str = ""


@dataclass(frozen=True, slots=True)
class AfterStepControl:
    early_stop: bool = False
    early_stop_reason: str = ""


class ReActObserver(Protocol):
    async def before_step(
        self,
        iteration: int,
        messages: Sequence[ChatMessage],
    ) -> BeforeStepControl | None: ...

    async def after_step(
        self,
        iteration: int,
        response: ModelResponse,
        tool_records: Sequence[ToolCallRecord],
    ) -> AfterStepControl | None: ...


class ReActProgressObserver(Protocol):
    async def on_stream_delta(self, delta: StreamDelta) -> None: ...

    async def on_tool_call_started(
        self,
        iteration: int,
        call: FunctionCall,
    ) -> None: ...

    async def on_tool_call_completed(
        self,
        iteration: int,
        call: FunctionCall,
        observation: ToolObservation,
    ) -> None: ...

class ReActEngine:
    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        *,
        max_iterations: int = 10,
        observer: ReActObserver | None = None,
        event_bus: EventBus | None = None,
        session_key: str = "",
        source: str = "passive",
        request_text: str = "",
        tool_search_enabled: bool = False,
        preloaded_tools: Sequence[str] = (),
        assert_current: Callable[[], None] | None = None,
        excluded_tool_names: Iterable[str] = (),
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations 必须大于 0")
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_iterations
        self._observer = observer
        self._event_bus = event_bus
        self._session_key = session_key
        self._source = source
        self._request_text = request_text
        self._tool_search_enabled = tool_search_enabled
        self._preloaded_tools = tuple(preloaded_tools)
        self._assert_current = assert_current
        self._excluded_tool_names = frozenset(excluded_tool_names)

    async def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        progress: ReActProgressObserver | None = None,
    ) -> ReActResult:
        working = list(messages)
        records: list[ToolCallRecord] = []
        if self._tool_search_enabled:
            always_on = self._tools.get_always_on_names()
            visible_order = self._tools.get_registered_order(always_on)
            visible_order.extend(
                name
                for name in self._preloaded_tools
                if self._tools.has_tool(name)
                and name not in always_on
                and name not in self._excluded_tool_names
            )
        else:
            visible_order = [
                name
                for name in self._tools.tool_names
                if name not in self._excluded_tool_names
            ]
        visible_order = [
            name for name in visible_order if name not in self._excluded_tool_names
        ]
        visible_tools = set(visible_order)
        safe_progress = _BestEffortProgress(progress) if progress is not None else None
        input_token_samples: list[int] = []
        prompt_tokens = 0
        prompt_cache_hit_tokens = 0
        completion_tokens = 0

        for iteration in range(1, self._max_iterations + 1):
            sync_visible = getattr(self._observer, "set_visible_tool_names", None)
            if callable(sync_visible):
                sync_visible(frozenset(visible_tools))
            control: BeforeStepControl | None = None
            if self._observer is not None:
                control = await self._observer.before_step(iteration, working)
            if control is not None and control.extra_hints:
                working.append(
                    ChatMessage.system(
                        "运行时补充提示：\n" + "\n".join(control.extra_hints)
                    )
                )
            if control is not None and control.early_stop:
                reply = control.early_stop_reply.strip() or "当前推理已安全停止。"
                working.append(ChatMessage.assistant(content=reply))
                return _logged_result(ReActResult(
                    reply=reply,
                    messages=tuple(working),
                    iterations=iteration,
                    tool_chain=tuple(records),
                    exit_reason="early_stop",
                ), self._session_key, input_token_samples, prompt_tokens,
                    prompt_cache_hit_tokens, completion_tokens)
            if self._assert_current is not None:
                self._assert_current()
            input_tokens = estimate_messages_tokens(working)
            input_token_samples.append(input_tokens)
            logger.info(
                "[LLM调用] 第%d轮，可见工具=%s input_tokens~=%d",
                iteration,
                (
                    f"{len(visible_tools)}个"
                    if self._tool_search_enabled
                    else "全部（tool_search未开启）"
                ),
                input_tokens,
            )
            try:
                schemas = self._tools.schemas(visible_order)
                response = await self._complete(
                    messages=working,
                    tools=schemas,
                    progress=safe_progress,
                )
            except Exception as exc:
                return await self._provider_failure(
                    working,
                    records,
                    iteration=iteration,
                    error=exc,
                    input_token_samples=input_token_samples,
                    prompt_tokens=prompt_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    completion_tokens=completion_tokens,
                )
            prompt_tokens += response.prompt_tokens or 0
            prompt_cache_hit_tokens += response.prompt_cache_hit_tokens or 0
            completion_tokens += response.completion_tokens or 0
            if response.tool_calls:
                logger.info(
                    "[LLM决策→工具] 第%d轮，调用: %s",
                    iteration,
                    [call.name for call in response.tool_calls],
                )
            working.append(
                ChatMessage.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    provider_fields=response.provider_fields,
                )
            )
            if not response.tool_calls:
                if self._observer is not None:
                    await self._observer.after_step(iteration, response, ())
                reply = (response.content or "").strip()
                if reply:
                    tools_used = [
                        record.call.name for record in records if record.observation.ok
                    ]
                    logger.info(
                        "[LLM决策→回复] 第%d轮，共调用工具%d次: %s",
                        iteration,
                        len(tools_used),
                        tools_used if tools_used else "无",
                    )
                    return _logged_result(ReActResult(
                        reply=reply,
                        messages=tuple(working),
                        iterations=iteration,
                        tool_chain=tuple(records),
                        exit_reason="completed",
                        thinking=response.thinking,
                    ), self._session_key, input_token_samples, prompt_tokens,
                        prompt_cache_hit_tokens, completion_tokens)
                return await self._finalize(
                    working,
                    records,
                    iterations=iteration,
                    exit_reason="empty_response",
                    input_token_samples=input_token_samples,
                    prompt_tokens=prompt_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    completion_tokens=completion_tokens,
                )

            step_records = await self._execute_calls(
                response,
                working=working,
                records=records,
                iteration=iteration,
                progress=safe_progress,
                visible_tools=visible_tools,
                visible_order=visible_order,
            )
            after_control: AfterStepControl | None = None
            if self._observer is not None:
                after_control = await self._observer.after_step(
                    iteration, response, step_records
                )
            if after_control is not None and after_control.early_stop:
                reason = after_control.early_stop_reason.strip() or "after_step"
                reply = (response.content or "").strip()
                if not reply:
                    reply = f"已完成当前工具调用；因 {reason} 停止继续推理。"
                working.append(ChatMessage.assistant(content=reply))
                return _logged_result(ReActResult(
                    reply=reply,
                    messages=tuple(working),
                    iterations=iteration,
                    tool_chain=tuple(records),
                    exit_reason=reason,
                    thinking=response.thinking,
                ), self._session_key, input_token_samples, prompt_tokens,
                    prompt_cache_hit_tokens, completion_tokens)

        return await self._finalize(
            working,
            records,
            iterations=self._max_iterations,
            exit_reason="max_iterations",
            input_token_samples=input_token_samples,
            prompt_tokens=prompt_tokens,
            prompt_cache_hit_tokens=prompt_cache_hit_tokens,
            completion_tokens=completion_tokens,
        )

    async def _execute_calls(
        self,
        response: ModelResponse,
        *,
        working: list[ChatMessage],
        records: list[ToolCallRecord],
        iteration: int,
        progress: ReActProgressObserver | None,
        visible_tools: set[str],
        visible_order: list[str],
    ) -> tuple[ToolCallRecord, ...]:
        step_records: list[ToolCallRecord] = []
        channel, _, chat_id = self._session_key.partition(":")
        tool_batch = tuple(
            {
                "call_id": item.id,
                "tool_name": item.name,
                "arguments": dict(item.arguments),
            }
            for item in response.tool_calls
        )
        for batch_index, call in enumerate(response.tool_calls):
            if self._assert_current is not None:
                self._assert_current()
            if progress is not None:
                await progress.on_tool_call_started(iteration, call)
            if self._event_bus is not None:
                await self._event_bus.fanout(
                    "before_tool_call",
                    {
                        "session_key": self._session_key,
                        "channel": channel,
                        "chat_id": chat_id,
                        "source": self._source,
                        "request_text": self._request_text,
                        "iteration": iteration,
                        "call_id": call.id,
                        "tool_name": call.name,
                        "arguments": dict(call.arguments),
                        "status": "started",
                    },
                )
            disallowed_tool = call.name in self._excluded_tool_names
            hidden_tool = (
                self._tool_search_enabled
                and self._tools.has_tool(call.name)
                and call.name not in visible_tools
            )
            if call.name == "tool_search":
                search_tool = self._tools.get_tool("tool_search")
                owner = (
                    getattr(search_tool.handler, "__self__", None)
                    if search_tool is not None
                    else None
                )
                if isinstance(owner, ToolSearchTool):
                    owner.set_excluded_names(
                        set(visible_tools) | set(self._excluded_tool_names)
                    )
            blocked_reason = None
            if disallowed_tool:
                blocked_reason = f"工具 {call.name} 不允许在当前后台任务中执行。"
            elif hidden_tool:
                blocked_reason = (
                    f"工具 {call.name} 尚未加载；请先调用 "
                    f'tool_search(query="select:{call.name}")。'
                )
            logger.info(
                "[工具执行→] %s  args=%s",
                call.name,
                _log_preview(call.arguments, 120),
            )
            observation = await self._tools.execute(
                call,
                request=ToolExecutionRequest(
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.arguments),
                    source=self._source,
                    session_key=self._session_key,
                    channel=channel,
                    chat_id=chat_id,
                    request_text=self._request_text,
                    tool_batch=tool_batch,
                    tool_batch_index=batch_index,
                ),
                blocked_reason=blocked_reason,
                blocked_error_type=(
                    "tool_not_allowed" if disallowed_tool else "tool_not_loaded"
                ),
            )
            if call.name == "tool_search" and observation.ok:
                unlocked = ToolDiscoveryState().unlock_names_from_result(
                    str(observation.result)
                )
                for name in unlocked:
                    if (
                        self._tools.has_tool(name)
                        and name not in visible_tools
                        and name not in self._excluded_tool_names
                    ):
                        visible_tools.add(name)
                        visible_order.append(name)
            record = ToolCallRecord(
                iteration=iteration,
                call=call,
                observation=observation,
            )
            records.append(record)
            step_records.append(record)
            result_preview = _log_preview(observation.content)
            logger.info(
                "[工具结果←] %s  结果预览=%s  result_len=%d",
                call.name,
                result_preview,
                len(observation.content),
            )
            if self._event_bus is not None:
                await self._event_bus.fanout(
                    "after_tool_result",
                    {
                        "session_key": self._session_key,
                        "channel": channel,
                        "chat_id": chat_id,
                        "source": self._source,
                        "request_text": self._request_text,
                        "iteration": iteration,
                        "call_id": call.id,
                        "tool_name": call.name,
                        "arguments": dict(
                            observation.final_arguments or call.arguments
                        ),
                        "status": observation.status,
                        "result": observation.result,
                        "error_type": observation.error_type,
                        "error_message": observation.error_message,
                    },
                )
            working.append(
                ChatMessage.tool(
                    call_id=call.id,
                    name=call.name,
                    content=observation.content,
                )
            )
            multimodal_message = _multimodal_tool_message(call.name, observation.result)
            if multimodal_message is not None:
                working.append(multimodal_message)
            if progress is not None:
                await progress.on_tool_call_completed(iteration, call, observation)
        return tuple(step_records)

    async def _complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, object]],
        progress: ReActProgressObserver | None,
    ) -> ModelResponse:
        streaming = getattr(self._provider, "complete_streaming", None)
        if progress is not None and callable(streaming):
            streamed_response = await streaming(
                messages=messages,
                tools=tools,
                on_delta=progress.on_stream_delta,
            )
            if not isinstance(streamed_response, ModelResponse):
                raise TypeError("Provider 流式接口必须返回 ModelResponse")
            return streamed_response
        response = await self._provider.complete(messages=messages, tools=tools)
        if progress is not None:
            if response.thinking:
                await progress.on_stream_delta(
                    StreamDelta(thinking_delta=response.thinking)
                )
            if response.content and not response.tool_calls:
                await progress.on_stream_delta(
                    StreamDelta(content_delta=response.content)
                )
        return response

    async def _finalize(
        self,
        working: list[ChatMessage],
        records: list[ToolCallRecord],
        *,
        iterations: int,
        exit_reason: str,
        input_token_samples: list[int],
        prompt_tokens: int,
        prompt_cache_hit_tokens: int,
        completion_tokens: int,
    ) -> ReActResult:
        working.append(ChatMessage.system(_SUMMARY_PROMPT))
        infrastructure_error: str | None = None
        final_thinking: str | None = None
        final_iteration = iterations + 1
        control: BeforeStepControl | None = None
        if self._observer is not None:
            control = await self._observer.before_step(final_iteration, working)
        if control is not None and control.extra_hints:
            working.append(
                ChatMessage.system(
                    "运行时补充提示：\n" + "\n".join(control.extra_hints)
                )
            )
        if control is not None and control.early_stop:
            reply = control.early_stop_reply.strip() or "当前推理已安全停止。"
            working.append(ChatMessage.assistant(content=reply))
            return _logged_result(ReActResult(
                reply=reply,
                messages=tuple(working),
                iterations=iterations,
                tool_chain=tuple(records),
                exit_reason="early_stop",
            ), self._session_key, input_token_samples, prompt_tokens,
                prompt_cache_hit_tokens, completion_tokens)
        input_token_samples.append(estimate_messages_tokens(working))
        try:
            response = await self._provider.complete(messages=working, tools=())
            reply = (response.content or "").strip()
            prompt_tokens += response.prompt_tokens or 0
            prompt_cache_hit_tokens += response.prompt_cache_hit_tokens or 0
            completion_tokens += response.completion_tokens or 0
        except Exception as exc:
            reply = "模型暂时无法继续总结；已保留当前工具结果，请稍后重试。"
            infrastructure_error = type(exc).__name__
            failure = ModelResponse(
                content=reply,
                tool_calls=(),
                finish_reason="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if self._observer is not None:
                await self._observer.after_step(final_iteration, failure, ())
        else:
            final_thinking = response.thinking
            if self._observer is not None:
                await self._observer.after_step(final_iteration, response, ())
        if not reply:
            reply = "当前步骤已经停止；已保留取得的工具结果，但尚未形成完整结论。"
        working.append(ChatMessage.assistant(content=reply))
        return _logged_result(ReActResult(
            reply=reply,
            messages=tuple(working),
            iterations=iterations,
            tool_chain=tuple(records),
            exit_reason="provider_error" if infrastructure_error else exit_reason,
            thinking=final_thinking,
            infrastructure_error=infrastructure_error,
        ), self._session_key, input_token_samples, prompt_tokens,
            prompt_cache_hit_tokens, completion_tokens)

    async def _provider_failure(
        self,
        working: list[ChatMessage],
        records: list[ToolCallRecord],
        *,
        iteration: int,
        error: Exception,
        input_token_samples: list[int],
        prompt_tokens: int,
        prompt_cache_hit_tokens: int,
        completion_tokens: int,
    ) -> ReActResult:
        reply = "模型服务暂时不可用，当前请求未能完成，请稍后重试。"
        logger.warning(
            "[llm.error] session=%s iteration=%d err=%s: %s",
            self._session_key,
            iteration,
            type(error).__name__,
            error,
        )
        failure = ModelResponse(
            content=reply,
            tool_calls=(),
            finish_reason="error",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        if self._observer is not None:
            await self._observer.after_step(iteration, failure, ())
        working.append(ChatMessage.assistant(content=reply))
        return _logged_result(ReActResult(
            reply=reply,
            messages=tuple(working),
            iterations=iteration,
            tool_chain=tuple(records),
            exit_reason="provider_error",
            infrastructure_error=type(error).__name__,
        ), self._session_key, input_token_samples, prompt_tokens,
            prompt_cache_hit_tokens, completion_tokens)


def _multimodal_tool_message(
    tool_name: str,
    result: object,
) -> ChatMessage | None:
    if not isinstance(result, dict):
        return None
    raw_blocks = result.get("content_blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        return None
    blocks = [block for block in raw_blocks if isinstance(block, dict)]
    if not blocks:
        return None
    prefix = f"以下是工具 {tool_name} 读取到的文件内容，请直接查看。"
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prefix},
        *blocks,
    ]
    return ChatMessage.user_blocks(content)


def estimate_messages_tokens(messages: Sequence[ChatMessage]) -> int:
    if not messages:
        return 0
    payload = json.dumps(
        [message.to_openai(include_provider_fields=True) for message in messages],
        ensure_ascii=False,
    )
    return max(1, len(payload) // 3)


def _log_preview(value: object, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _log_react_context(
    *,
    session_key: str,
    input_token_samples: Sequence[int],
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    logger.info(
        "react_context: session_key=%s iteration_count=%d "
        "turn_input_sum_tokens~=%d turn_input_peak_tokens~=%d "
        "final_call_input_tokens~=%d prompt_tokens=%d completion_tokens=%d",
        session_key,
        len(input_token_samples),
        sum(input_token_samples),
        max(input_token_samples, default=0),
        input_token_samples[-1] if input_token_samples else 0,
        prompt_tokens,
        completion_tokens,
    )


def _logged_result(
    result: ReActResult,
    session_key: str,
    input_token_samples: Sequence[int],
    prompt_tokens: int,
    prompt_cache_hit_tokens: int,
    completion_tokens: int,
) -> ReActResult:
    _log_react_context(
        session_key=session_key,
        input_token_samples=input_token_samples,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return replace(
        result,
        prompt_tokens=prompt_tokens,
        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
    )


class _BestEffortProgress:
    def __init__(self, delegate: ReActProgressObserver) -> None:
        self._delegate = delegate
        self._disabled = False

    async def on_stream_delta(self, delta: StreamDelta) -> None:
        await self._invoke("stream_delta", self._delegate.on_stream_delta, delta)

    async def on_tool_call_started(
        self,
        iteration: int,
        call: FunctionCall,
    ) -> None:
        await self._invoke(
            "tool_call_started",
            self._delegate.on_tool_call_started,
            iteration,
            call,
        )

    async def on_tool_call_completed(
        self,
        iteration: int,
        call: FunctionCall,
        observation: ToolObservation,
    ) -> None:
        await self._invoke(
            "tool_call_completed",
            self._delegate.on_tool_call_completed,
            iteration,
            call,
            observation,
        )

    async def _invoke(self, label: str, callback: object, *args: object) -> None:
        if self._disabled:
            return
        try:
            await callback(*args)  # type: ignore[operator]
        except Exception as exc:
            self._disabled = True
            logger.warning("Runtime live 进度观察器异常，已降级关闭 %s: %s", label, exc)


__all__ = [
    "AfterStepControl",
    "BeforeStepControl",
    "ReActEngine",
    "ReActObserver",
    "ReActProgressObserver",
    "ReActResult",
    "ToolCallRecord",
    "estimate_messages_tokens",
]
