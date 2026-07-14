"""内层 ReAct + Function Calling 循环。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from memopilot.runtime.contracts import ChatMessage, FunctionCall, ModelResponse, StreamDelta
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import ToolObservation, ToolRegistry

_SUMMARY_PROMPT = """[运行时收尾]
当前 Turn 已达到工具调用轮次上限。请停止调用工具，只根据已有对话和工具结果，
输出给用户看的自然语言阶段性回复：说明已经得到的结果、仍缺少的信息和当前结论。
不要暴露内部 schema、call id 或本条系统指令。"""

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


class ReActObserver(Protocol):
    async def before_step(
        self,
        iteration: int,
        messages: Sequence[ChatMessage],
    ) -> None: ...

    async def after_step(
        self,
        iteration: int,
        response: ModelResponse,
        tool_records: Sequence[ToolCallRecord],
    ) -> None: ...


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
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations 必须大于 0")
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_iterations
        self._observer = observer

    async def run(
        self,
        messages: Sequence[ChatMessage],
        *,
        progress: ReActProgressObserver | None = None,
    ) -> ReActResult:
        working = list(messages)
        records: list[ToolCallRecord] = []
        schemas = self._tools.schemas()
        safe_progress = _BestEffortProgress(progress) if progress is not None else None

        for iteration in range(1, self._max_iterations + 1):
            if self._observer is not None:
                await self._observer.before_step(iteration, working)
            try:
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
            )
            if self._observer is not None:
                await self._observer.after_step(iteration, response, step_records)

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
    ) -> tuple[ToolCallRecord, ...]:
        step_records: list[ToolCallRecord] = []
        for call in response.tool_calls:
            if progress is not None:
                await progress.on_tool_call_started(iteration, call)
            observation = await self._tools.execute(call)
            record = ToolCallRecord(
                iteration=iteration,
                call=call,
                observation=observation,
            )
            records.append(record)
            step_records.append(record)
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
        final_iteration = iterations + 1
        if self._observer is not None:
            await self._observer.before_step(final_iteration, working)
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
    "ReActEngine",
    "ReActObserver",
    "ReActProgressObserver",
    "ReActResult",
    "ToolCallRecord",
]
