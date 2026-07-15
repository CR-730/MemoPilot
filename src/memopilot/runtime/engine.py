"""串联外层 Phase Pipeline 与内层 ReAct 的 Turn Runtime。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, cast

from memopilot.extensions.events import EventBus
from memopilot.extensions.prompts import PromptBlock, PromptRenderer
from memopilot.extensions.skills import SkillCatalog
from memopilot.memory.contracts import MemoryQuery, MemoryQueryEngine
from memopilot.runtime.contracts import ChatMessage, ModelResponse
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
    prompt_scope: str = "passive"


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
        prompt_blocks: Sequence[PromptBlock] = (),
        prompt_max_chars: int = 12000,
        event_bus: EventBus | None = None,
        skills: SkillCatalog | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_iterations
        self._memory_engine = memory_engine
        self._memory_profile = memory_profile
        self._event_bus = event_bus
        prompt_renderer = PromptRenderer(tuple(prompt_blocks))
        if prompt_max_chars <= 0:
            raise ValueError("Prompt 总预算必须大于 0")
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
                        ("memory.context",),
                        self._memory_prerecall,
                    ),
                ),
            )
        all_modules = cast(
            Sequence[PhaseModule],
            (
                *_default_modules(prompt_renderer, prompt_max_chars, skills),
                *memory_modules,
                *modules,
            ),
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
        parts: list[str] = []
        if self._memory_profile is not None:
            for name, title in (
                ("MEMORY.md", "用户长期记忆"),
                ("SELF.md", "助手自我认知"),
                ("CONTEXT.md", "近期上下文"),
            ):
                content = self._memory_profile.read(name).strip()
                if content:
                    parts.append(f"## {title}\n{content}")
        if self._memory_engine is not None:
            result = await self._memory_engine.query(
                MemoryQuery(
                    text=turn.content,
                    intent="context",
                    session_key=turn.session_key,
                )
            )
            if result.text_block.strip():
                parts.append(result.text_block.strip())
        context_block = "\n\n".join(parts)
        return {"memory.context": context_block[: self._memory_markdown_max_chars]}

    async def run(
        self,
        turn: TurnInput,
        *,
        step_sink: RuntimeStepSink | None = None,
        progress: ReActProgressObserver | None = None,
    ) -> TurnResult:
        context = PhaseContext(slots={"turn.input": turn})
        execution = _RuntimeExecution(
            self._pipeline,
            context,
            step_sink,
            event_bus=self._event_bus,
        )
        await execution.run_phase(LifecyclePhase.BEFORE_TURN)
        await execution.run_phase(LifecyclePhase.BEFORE_REASONING)
        await execution.run_phase(LifecyclePhase.PROMPT_RENDER)
        prompt_messages = context.slots["prompt.messages"]
        if not isinstance(prompt_messages, tuple):
            raise RuntimeError("PromptRender 未产生 tuple[ChatMessage, ...]")

        react = await ReActEngine(
            self._provider,
            self._tools,
            max_iterations=self._max_iterations,
            observer=execution,
        ).run(prompt_messages, progress=progress)
        context.slots["reasoning.result"] = react
        await execution.run_phase(LifecyclePhase.AFTER_REASONING)
        await execution.run_phase(LifecyclePhase.AFTER_TURN)
        return TurnResult(
            reply=react.reply,
            messages=react.messages,
            react=react,
            phase_trace=tuple(execution.phase_trace),
            trace=tuple(execution.trace),
        )


class _RuntimeExecution(ReActObserver):
    def __init__(
        self,
        pipeline: PhasePipeline,
        context: PhaseContext,
        sink: RuntimeStepSink | None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._context = context
        self._sink = sink
        self._event_bus = event_bus
        self.phase_trace: list[PhaseTraceEntry] = []
        self.trace: list[RuntimeTraceEvent] = []
        self._iteration: int | None = None

    async def run_phase(self, phase: LifecyclePhase) -> None:
        if self._event_bus is not None:
            updated = await self._event_bus.emit(phase.value, self._context.slots)
            self._context.slots.update(updated)
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
                        "original_arguments": record.observation.original_arguments,
                        "final_arguments": record.observation.final_arguments,
                        "hook_trace": record.observation.hook_trace,
                    },
                )
            )
        await self.run_phase(LifecyclePhase.AFTER_STEP)

    async def _record(self, event: RuntimeTraceEvent) -> None:
        self.trace.append(event)
        if self._sink is not None:
            await self._sink.record(event)


async def _before_turn(context: PhaseContext) -> Mapping[str, Any]:
    turn = context.slots["turn.input"]
    if not isinstance(turn, TurnInput):
        raise TypeError("turn.input 必须是 TurnInput")
    if not turn.session_key or not turn.content.strip():
        raise ValueError("session_key 和用户消息不能为空")
    return {"turn.request": turn}


async def _before_reasoning(context: PhaseContext) -> Mapping[str, Any]:
    return {"reasoning.input": context.slots["turn.request"]}


async def _prompt_render(
    context: PhaseContext,
    *,
    renderer: PromptRenderer,
    max_chars: int,
    skills: SkillCatalog | None,
) -> Mapping[str, Any]:
    turn = context.slots["reasoning.input"]
    if not isinstance(turn, TurnInput):
        raise TypeError("reasoning.input 必须是 TurnInput")
    messages: list[ChatMessage] = []
    system_parts = [turn.system_prompt.strip()]
    memory_context = context.slots.get("memory.context")
    if isinstance(memory_context, str) and memory_context.strip():
        system_parts.append("以下是与当前问题相关的长期记忆：\n" + memory_context.strip())
    if skills is not None:
        skill_prompt = skills.render_mentions(turn.content, max_chars=max_chars // 2)
        if skill_prompt:
            system_parts.append(skill_prompt)
    system_prompt = renderer.render(
        scope=turn.prompt_scope,
        base_prompt="\n\n".join(part for part in system_parts if part),
        max_chars=max_chars,
    )
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


def _default_modules(
    prompt_renderer: PromptRenderer,
    prompt_max_chars: int,
    skills: SkillCatalog | None,
) -> tuple[FunctionPhaseModule, ...]:
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
            partial(
                _prompt_render,
                renderer=prompt_renderer,
                max_chars=prompt_max_chars,
                skills=skills,
            ),
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
