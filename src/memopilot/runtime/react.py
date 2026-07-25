"""内层 ReAct + Function Calling 循环。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

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
                return ReActResult(
                    reply=reply,
                    messages=tuple(working),
                    iterations=iteration,
                    tool_chain=tuple(records),
                    exit_reason="early_stop",
                )
            if self._assert_current is not None:
                self._assert_current()
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
                    return ReActResult(
                        reply=reply,
                        messages=tuple(working),
                        iterations=iteration,
                        tool_chain=tuple(records),
                        exit_reason="completed",
                        thinking=response.thinking,
                    )
                return await self._finalize(
                    working,
                    records,
                    iterations=iteration,
                    exit_reason="empty_response",
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
                return ReActResult(
                    reply=reply,
                    messages=tuple(working),
                    iterations=iteration,
                    tool_chain=tuple(records),
                    exit_reason=reason,
                    thinking=response.thinking,
                )

        return await self._finalize(
            working,
            records,
            iterations=self._max_iterations,
            exit_reason="max_iterations",
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
            return ReActResult(
                reply=reply,
                messages=tuple(working),
                iterations=iterations,
                tool_chain=tuple(records),
                exit_reason="early_stop",
            )
        try:
            response = await self._provider.complete(messages=working, tools=())
            reply = (response.content or "").strip()
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
        return ReActResult(
            reply=reply,
            messages=tuple(working),
            iterations=iterations,
            tool_chain=tuple(records),
            exit_reason="provider_error" if infrastructure_error else exit_reason,
            thinking=final_thinking,
            infrastructure_error=infrastructure_error,
        )

    async def _provider_failure(
        self,
        working: list[ChatMessage],
        records: list[ToolCallRecord],
        *,
        iteration: int,
        error: Exception,
    ) -> ReActResult:
        reply = "模型服务暂时不可用，当前请求未能完成，请稍后重试。"
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
        return ReActResult(
            reply=reply,
            messages=tuple(working),
            iterations=iteration,
            tool_chain=tuple(records),
            exit_reason="provider_error",
            infrastructure_error=type(error).__name__,
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
]
