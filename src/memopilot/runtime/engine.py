"""串联外层 Phase Pipeline 与内层 ReAct 的 Turn Runtime。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from memopilot.memory.contracts import MemoryQuery, MemoryQueryEngine
from memopilot.memory.tool_context import (
    bind_memory_tool_context,
    reset_memory_tool_context,
)
from memopilot.runtime.contracts import ChatMessage, ModelResponse
from memopilot.runtime.memory_citations import CITATION_PROTOCOL, extract_cited_ids
from memopilot.runtime.phases import (
    LifecyclePhase,
    PhaseContext,
    PhaseModule,
    PhasePipeline,
)
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.react import (
    ReActEngine,
    ReActObserver,
    ReActProgressObserver,
    ReActResult,
    ToolCallRecord,
)
from memopilot.runtime.tools import ToolRegistry


@dataclass(frozen=True)
class TurnInput:
    session_key: str
    content: str
    system_prompt: str = ""
    history: tuple[ChatMessage, ...] = ()
    current_user_content: str | None = None
    resume_snapshot_id: str | None = None
    interrupt_original_message: str | None = None
    memory_source_ref: str = ""


@dataclass(frozen=True)
class PhaseTraceEntry:
    phase: LifecyclePhase
    module_slot: str
    iteration: int | None


@dataclass(frozen=True)
class RuntimeTraceEvent:
    phase: LifecyclePhase
    step_type: str
    state: str = "succeeded"
    iteration: int | None = None
    module_slot: str | None = None
    tool_name: str | None = None
    input: Mapping[str, Any] | None = None
    observation: Mapping[str, Any] | None = None


class RuntimeStepSink(Protocol):
    async def record(self, event: RuntimeTraceEvent) -> None: ...


class LongTermMemoryProfile(Protocol):
    def read(self, name: str) -> str: ...


@dataclass(frozen=True)
class TurnResult:
    reply: str
    messages: tuple[ChatMessage, ...]
    react: ReActResult
    phase_trace: tuple[PhaseTraceEntry, ...]
    trace: tuple[RuntimeTraceEvent, ...]
    cited_memory_ids: tuple[str, ...] = ()


PhaseHandler = Callable[[PhaseContext], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class FunctionPhaseModule:
    phase: LifecyclePhase
    slot: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    handler: PhaseHandler

    async def run(self, context: PhaseContext) -> Mapping[str, Any]:
        return await self.handler(context)


class AgentRuntime:
    def __init__(
        self,
        provider: ChatProvider,
        tools: ToolRegistry,
        *,
        max_iterations: int = 10,
        modules: Sequence[PhaseModule] = (),
        memory_engine: MemoryQueryEngine | None = None,
        memory_profile: LongTermMemoryProfile | None = None,
        memory_markdown_max_chars: int = 6000,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_iterations
        self._memory_engine = memory_engine
        self._memory_profile = memory_profile
        if memory_markdown_max_chars < 500:
            raise ValueError("Markdown 记忆注入预算不能少于 500 字符")
        self._memory_markdown_max_chars = memory_markdown_max_chars
        memory_modules: tuple[PhaseModule, ...] = ()
        if memory_engine is not None or memory_profile is not None:
            memory_modules = cast(
                tuple[PhaseModule, ...],
                (
                    FunctionPhaseModule(
                        LifecyclePhase.BEFORE_REASONING,
                        "before_reasoning.memory_prerecall",
                        ("reasoning.input",),
                        ("memory.context", "memory.trace"),
                        self._memory_prerecall,
                    ),
                ),
            )
        all_modules = cast(
            Sequence[PhaseModule],
            (*_default_modules(), *memory_modules, *modules),
        )
        self._pipeline = PhasePipeline(
            all_modules,
            initial_slots={"turn.input"},
            provided_slots={
                LifecyclePhase.BEFORE_STEP: {
                    "step.iteration",
                    "step.messages",
                },
                LifecyclePhase.AFTER_STEP: {"step.response"},
                LifecyclePhase.AFTER_REASONING: {"reasoning.result"},
            },
        )

    async def _memory_prerecall(self, context: PhaseContext) -> Mapping[str, Any]:
        turn = context.slots["reasoning.input"]
        if not isinstance(turn, TurnInput):
            raise TypeError("reasoning.input 必须是 TurnInput")
        sections: dict[str, str] = {}
        memory_trace: dict[str, object] = {}
        if self._memory_profile is not None:
            for name, key, title in (
                ("MEMORY.md", "memory", "用户长期记忆"),
                ("SELF.md", "self", "助手自我认知"),
                ("RECENT_CONTEXT.md", "recent", "近期上下文"),
            ):
                content = self._memory_profile.read(name).strip()
                if name == "RECENT_CONTEXT.md":
                    content = _without_recent_turns(content)
                if content:
                    sections[key] = f"## {title}\n{content}"
        if self._memory_engine is not None:
            result = await self._memory_engine.query(
                MemoryQuery(
                    text=turn.content,
                    intent="context",
                    session_key=turn.session_key,
                )
            )
            if result.text_block.strip():
                sections["retrieval"] = result.text_block.strip()
            memory_trace = dict(result.trace)
        selected = _fit_memory_sections(sections, self._memory_markdown_max_chars)
        context_block = "\n\n".join(
            selected[key]
            for key in ("memory", "self", "recent", "retrieval")
            if selected.get(key)
        )
        injected = memory_trace.get("injected_ids")
        if isinstance(injected, list):
            retrieval_text = selected.get("retrieval", "")
            memory_trace["injected_ids"] = [
                str(item_id)
                for item_id in injected
                if f"[{item_id}]" in retrieval_text
            ]
        return {
            "memory.context": context_block,
            "memory.trace": memory_trace,
        }

    async def run(
        self,
        turn: TurnInput,
        *,
        step_sink: RuntimeStepSink | None = None,
        progress: ReActProgressObserver | None = None,
        memory_assert_current: Callable[[], None] | None = None,
        memory_fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> TurnResult:
        context = PhaseContext(slots={"turn.input": turn})
        execution = _RuntimeExecution(self._pipeline, context, step_sink)
        await execution.run_phase(LifecyclePhase.BEFORE_TURN)
        await execution.run_phase(LifecyclePhase.BEFORE_REASONING)
        memory_trace = context.slots.get("memory.trace")
        if isinstance(memory_trace, dict) and memory_trace:
            await execution.record_event(
                RuntimeTraceEvent(
                    phase=LifecyclePhase.BEFORE_REASONING,
                    step_type="memory_recall",
                    observation=memory_trace,
                )
            )
        await execution.run_phase(LifecyclePhase.PROMPT_RENDER)
        prompt_messages = context.slots["prompt.messages"]
        if not isinstance(prompt_messages, tuple):
            raise RuntimeError("PromptRender 未产生 tuple[ChatMessage, ...]")

        tool_context_token = bind_memory_tool_context(
            turn.session_key,
            source_ref=turn.memory_source_ref,
            assert_current=memory_assert_current,
            fenced_write=memory_fenced_write,
        )
        try:
            react = await ReActEngine(
                self._provider,
                self._tools,
                max_iterations=self._max_iterations,
                observer=execution,
            ).run(prompt_messages, progress=progress)
        finally:
            reset_memory_tool_context(tool_context_token)
        cleaned_reply, cited_memory_ids = extract_cited_ids(react.reply)
        allowed_memory_ids = _allowed_memory_ids(context, react)
        cited_memory_ids = tuple(
            item_id for item_id in cited_memory_ids if item_id in allowed_memory_ids
        )
        if cleaned_reply != react.reply:
            messages = list(react.messages)
            if messages and messages[-1].role == "assistant":
                messages[-1] = replace(messages[-1], content=cleaned_reply)
            react = replace(react, reply=cleaned_reply, messages=tuple(messages))
        context.slots["reasoning.result"] = react
        await execution.run_phase(LifecyclePhase.AFTER_REASONING)
        await execution.run_phase(LifecyclePhase.AFTER_TURN)
        return TurnResult(
            reply=react.reply,
            messages=react.messages,
            react=react,
            phase_trace=tuple(execution.phase_trace),
            trace=tuple(execution.trace),
            cited_memory_ids=cited_memory_ids,
        )


class _RuntimeExecution(ReActObserver):
    def __init__(
        self,
        pipeline: PhasePipeline,
        context: PhaseContext,
        sink: RuntimeStepSink | None,
    ) -> None:
        self._pipeline = pipeline
        self._context = context
        self._sink = sink
        self.phase_trace: list[PhaseTraceEntry] = []
        self.trace: list[RuntimeTraceEvent] = []
        self._iteration: int | None = None

    async def run_phase(self, phase: LifecyclePhase) -> None:
        trace_start = len(self._context.trace)
        await self._pipeline.run_phase(phase, self._context)
        for traced_phase, module_slot in self._context.trace[trace_start:]:
            entry = PhaseTraceEntry(traced_phase, module_slot, self._iteration)
            self.phase_trace.append(entry)
            await self._record(
                RuntimeTraceEvent(
                    phase=traced_phase,
                    step_type="phase_module",
                    iteration=self._iteration,
                    module_slot=module_slot,
                )
            )

    async def before_step(
        self,
        iteration: int,
        messages: Sequence[ChatMessage],
    ) -> None:
        self._iteration = iteration
        self._context.slots["step.iteration"] = iteration
        self._context.slots["step.messages"] = tuple(messages)
        await self.run_phase(LifecyclePhase.BEFORE_STEP)

    async def after_step(
        self,
        iteration: int,
        response: ModelResponse,
        tool_records: Sequence[ToolCallRecord],
    ) -> None:
        self._iteration = iteration
        self._context.slots["step.response"] = response
        await self._record(
            RuntimeTraceEvent(
                phase=LifecyclePhase.AFTER_STEP,
                step_type="model",
                state="failed" if response.error_type else "succeeded",
                iteration=iteration,
                input={"tool_count": len(response.tool_calls)},
                observation={
                    "finish_reason": response.finish_reason,
                    "response_id": response.response_id,
                    "has_content": bool(response.content),
                    "error_type": response.error_type,
                    "error_message": response.error_message,
                },
            )
        )
        for record in tool_records:
            await self._record(
                RuntimeTraceEvent(
                    phase=LifecyclePhase.AFTER_STEP,
                    step_type="tool",
                    state="succeeded" if record.observation.ok else "failed",
                    iteration=iteration,
                    tool_name=record.call.name,
                    input=record.call.arguments,
                    observation={
                        "ok": record.observation.ok,
                        "result": record.observation.result,
                        "error_type": record.observation.error_type,
                        "error_message": record.observation.error_message,
                    },
                )
            )
        await self.run_phase(LifecyclePhase.AFTER_STEP)

    async def _record(self, event: RuntimeTraceEvent) -> None:
        self.trace.append(event)
        if self._sink is not None:
            await self._sink.record(event)

    async def record_event(self, event: RuntimeTraceEvent) -> None:
        await self._record(event)


async def _before_turn(context: PhaseContext) -> Mapping[str, Any]:
    turn = context.slots["turn.input"]
    if not isinstance(turn, TurnInput):
        raise TypeError("turn.input 必须是 TurnInput")
    if not turn.session_key or not turn.content.strip():
        raise ValueError("session_key 和用户消息不能为空")
    return {"turn.request": turn}


async def _before_reasoning(context: PhaseContext) -> Mapping[str, Any]:
    return {"reasoning.input": context.slots["turn.request"]}


async def _prompt_render(context: PhaseContext) -> Mapping[str, Any]:
    turn = context.slots["reasoning.input"]
    if not isinstance(turn, TurnInput):
        raise TypeError("reasoning.input 必须是 TurnInput")
    messages: list[ChatMessage] = []
    system_parts = [turn.system_prompt.strip()]
    memory_context = context.slots.get("memory.context")
    if isinstance(memory_context, str) and memory_context.strip():
        system_parts.append("以下是与当前问题相关的长期记忆：\n" + memory_context.strip())
        system_parts.append(CITATION_PROTOCOL)
    system_prompt = "\n\n".join(part for part in system_parts if part)
    if system_prompt:
        messages.append(ChatMessage.system(system_prompt))
    messages.extend(turn.history)
    messages.append(ChatMessage.user(turn.content))
    return {"prompt.messages": tuple(messages)}


async def _before_step(context: PhaseContext) -> Mapping[str, Any]:
    return {"step.ready": context.slots["step.iteration"]}


async def _after_step(context: PhaseContext) -> Mapping[str, Any]:
    return {"step.observed": context.slots["step.response"]}


async def _after_reasoning(context: PhaseContext) -> Mapping[str, Any]:
    return {"turn.output": context.slots["reasoning.result"]}


async def _after_turn(context: PhaseContext) -> Mapping[str, Any]:
    return {"turn.completed": context.slots["turn.output"] is not None}


def _default_modules() -> tuple[FunctionPhaseModule, ...]:
    return (
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_TURN,
            "before_turn.validate",
            ("turn.input",),
            ("turn.request",),
            _before_turn,
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_REASONING,
            "before_reasoning.prepare",
            ("turn.request",),
            ("reasoning.input",),
            _before_reasoning,
        ),
        FunctionPhaseModule(
            LifecyclePhase.PROMPT_RENDER,
            "prompt_render.messages",
            ("reasoning.input",),
            ("prompt.messages",),
            _prompt_render,
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_STEP,
            "before_step.prepare",
            ("prompt.messages", "step.iteration", "step.messages"),
            ("step.ready",),
            _before_step,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_STEP,
            "after_step.observe",
            ("step.ready", "step.response"),
            ("step.observed",),
            _after_step,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_REASONING,
            "after_reasoning.finalize",
            ("reasoning.result",),
            ("turn.output",),
            _after_reasoning,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_TURN,
            "after_turn.complete",
            ("turn.output",),
            ("turn.completed",),
            _after_turn,
        ),
    )


def _allowed_memory_ids(context: PhaseContext, react: ReActResult) -> set[str]:
    allowed: set[str] = set()
    trace = context.slots.get("memory.trace")
    if isinstance(trace, dict):
        values = trace.get("injected_ids")
        if isinstance(values, list):
            allowed.update(str(value) for value in values if str(value).strip())
    for record in react.tool_chain:
        if record.call.name != "recall_memory" or not record.observation.ok:
            continue
        result = record.observation.result
        if not isinstance(result, dict):
            continue
        values = result.get("cited_item_ids")
        if isinstance(values, list):
            allowed.update(str(value) for value in values if str(value).strip())
    return allowed


def _fit_memory_sections(sections: Mapping[str, str], budget: int) -> dict[str, str]:
    """先保留本轮检索和近期上下文，再用剩余预算填长期 Markdown。"""
    selected: dict[str, str] = {}
    remaining = budget
    for key in ("retrieval", "recent", "self", "memory"):
        value = sections.get(key, "").strip()
        if not value or remaining <= 0:
            continue
        separator_cost = 2 if selected else 0
        if remaining <= separator_cost:
            break
        kept = value[: remaining - separator_cost].rstrip()
        if kept:
            selected[key] = kept
            remaining -= len(kept) + separator_cost
    return selected


def _without_recent_turns(content: str) -> str:
    marker = "\n## Recent Turns\n"
    if marker not in content:
        return content.strip()
    return content.split(marker, 1)[0].rstrip()


__all__ = [
    "AgentRuntime",
    "FunctionPhaseModule",
    "LongTermMemoryProfile",
    "PhaseTraceEntry",
    "RuntimeStepSink",
    "RuntimeTraceEvent",
    "TurnInput",
    "TurnResult",
]
