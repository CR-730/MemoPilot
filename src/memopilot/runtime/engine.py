"""串联外层 Phase Pipeline 与内层 ReAct 的 Turn Runtime。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import Any, Protocol, cast

from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_events import (
    AfterReasoningCtx,
    AfterReasoningInput,
    AfterReasoningResult,
    AfterStepCtx,
    AfterTurnCtx,
    AfterTurnInput,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeStepCtx,
    BeforeStepInput,
    BeforeTurnCtx,
    BeforeTurnInput,
    PromptRenderInput,
    PromptRenderResult,
)
from memopilot.extensions.prompts import (
    PromptBlock,
    PromptRenderContext,
    PromptRenderer,
)
from memopilot.extensions.skills import ActiveSkillBlock, SkillCatalog
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
    AfterStepControl,
    BeforeStepControl,
    ReActEngine,
    ReActObserver,
    ReActProgressObserver,
    ReActResult,
    ToolCallRecord,
)
from memopilot.runtime.tool_search import ToolDiscoveryState, format_deferred_tools_hint
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
    media: tuple[str, ...] = ()
    outbound_metadata: Mapping[str, object] = field(default_factory=dict)
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
    media: tuple[str, ...] = ()
    outbound_metadata: Mapping[str, object] = field(default_factory=dict)
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
        prompt_blocks: Sequence[PromptBlock] = (),
        prompt_max_chars: int = 12000,
        event_bus: EventBus | None = None,
        skills: SkillCatalog | None = None,
        tool_search_enabled: bool = False,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_iterations
        self._memory_engine = memory_engine
        self._memory_profile = memory_profile
        self._event_bus = event_bus
        self._tool_search_enabled = tool_search_enabled
        self._tool_discovery = ToolDiscoveryState()
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
                        ("memory.context", "memory.trace"),
                        self._memory_prerecall,
                    ),
                ),
            )
        all_modules = cast(
            Sequence[PhaseModule],
            (
                *_default_modules(
                    prompt_renderer,
                    prompt_max_chars,
                    skills,
                    event_bus,
                ),
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
        state = _turn_state(turn)
        context = PhaseContext(
            slots={
                "turn.input": turn,
                "before_turn.input": state,
            }
        )
        execution = _RuntimeExecution(
            self._pipeline,
            context,
            step_sink,
            event_bus=self._event_bus,
            tools=self._tools,
        )
        await execution.run_phase(LifecyclePhase.BEFORE_TURN)
        before_turn_ctx = cast(BeforeTurnCtx, context.slots["session:ctx"])
        if before_turn_ctx.abort:
            return _aborted_turn_result(
                turn,
                before_turn_ctx,
                exit_reason="before_turn_abort",
                execution=execution,
            )
        await execution.run_phase(LifecyclePhase.BEFORE_REASONING)
        before_reasoning_ctx = cast(
            BeforeReasoningCtx, context.slots["reasoning:ctx"]
        )
        if before_reasoning_ctx.abort:
            return _aborted_turn_result(
                cast(TurnInput, context.slots["turn.input"]),
                before_reasoning_ctx,
                exit_reason="before_reasoning_abort",
                execution=execution,
            )
        preloaded_tools = self._tool_discovery.get_preloaded_ordered(
            before_reasoning_ctx.session_key
        )
        if self._tool_search_enabled:
            visible = self._tools.get_always_on_names() | set(preloaded_tools)
            before_reasoning_ctx.visible_tool_names = frozenset(visible)
            deferred_hint = format_deferred_tools_hint(
                self._tools.get_deferred_names(visible)
            )
            if deferred_hint:
                before_reasoning_ctx.extra_hints.append(deferred_hint)
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
        effective_turn = context.slots.get("turn.input")
        if not isinstance(effective_turn, TurnInput):
            raise RuntimeError("BeforeTurn 未保留 TurnInput")

        tool_context_token = bind_memory_tool_context(
            effective_turn.session_key,
            source_ref=effective_turn.memory_source_ref,
            assert_current=memory_assert_current,
            fenced_write=memory_fenced_write,
        )
        try:
            react = await ReActEngine(
                self._provider,
                self._tools,
                max_iterations=self._max_iterations,
                observer=execution,
                event_bus=self._event_bus,
                session_key=effective_turn.session_key,
                source=effective_turn.prompt_scope,
                request_text=effective_turn.content,
                tool_search_enabled=self._tool_search_enabled,
                preloaded_tools=preloaded_tools,
            ).run(prompt_messages, progress=progress)
        finally:
            reset_memory_tool_context(tool_context_token)
        if self._tool_search_enabled:
            self._tool_discovery.update(
                effective_turn.session_key,
                [
                    record.call.name
                    for record in react.tool_chain
                    if record.observation.status == "success"
                ],
                self._tools.get_always_on_names(),
            )
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
        context.slots["after_reasoning.input"] = AfterReasoningInput(
            cast(BeforeTurnInput, context.slots["before_turn.input"]),
            react,
        )
        await execution.run_phase(LifecyclePhase.AFTER_REASONING)
        await execution.run_phase(LifecyclePhase.AFTER_TURN)
        final_react = context.slots.get("turn.output")
        if not isinstance(final_react, ReActResult):
            raise RuntimeError("AfterReasoning 未产生 ReActResult")
        return TurnResult(
            reply=final_react.reply,
            messages=final_react.messages,
            react=final_react,
            phase_trace=tuple(execution.phase_trace),
            trace=tuple(execution.trace),
            media=tuple(
                cast(AfterReasoningCtx, context.slots["reasoning:ctx"]).media
            ),
            outbound_metadata=dict(
                cast(
                    AfterReasoningCtx,
                    context.slots["reasoning:ctx"],
                ).outbound_metadata
            ),
            cited_memory_ids=cited_memory_ids,
        )


def _aborted_turn_result(
    turn: TurnInput,
    ctx: BeforeTurnCtx | BeforeReasoningCtx,
    *,
    exit_reason: str,
    execution: _RuntimeExecution,
) -> TurnResult:
    reply = ctx.abort_reply.strip() or "当前请求已停止。"
    messages = (
        *ctx.history,
        ChatMessage.user(ctx.content),
        ChatMessage.assistant(content=reply),
    )
    react = ReActResult(
        reply=reply,
        messages=messages,
        iterations=0,
        tool_chain=(),
        exit_reason=exit_reason,
    )
    outbound_metadata = getattr(ctx, "outbound_metadata", turn.outbound_metadata)
    return TurnResult(
        reply=reply,
        messages=messages,
        react=react,
        phase_trace=tuple(execution.phase_trace),
        trace=tuple(execution.trace),
        media=ctx.media,
        outbound_metadata=dict(outbound_metadata),
    )


class _RuntimeExecution(ReActObserver):
    def __init__(
        self,
        pipeline: PhasePipeline,
        context: PhaseContext,
        sink: RuntimeStepSink | None,
        *,
        event_bus: EventBus | None = None,
        tools: ToolRegistry,
    ) -> None:
        self._pipeline = pipeline
        self._context = context
        self._sink = sink
        self._event_bus = event_bus
        self._tools = tools
        self.phase_trace: list[PhaseTraceEntry] = []
        self.trace: list[RuntimeTraceEvent] = []
        self._iteration: int | None = None

    async def run_phase(self, phase: LifecyclePhase) -> None:
        self._prepare_phase_contract(phase)
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
        if self._event_bus is not None and phase in {
            LifecyclePhase.AFTER_STEP,
            LifecyclePhase.AFTER_TURN,
        }:
            await self._event_bus.fanout(phase.value, self._phase_event_payload())

    def _phase_event_payload(self) -> dict[str, object]:
        payload: dict[str, object] = dict(self._context.slots)
        ctx_slot = _phase_ctx_slot(self._current_phase())
        ctx = self._context.slots.get(ctx_slot)
        if ctx is not None:
            try:
                payload.update(asdict(ctx))
            except TypeError:
                pass
        turn = self._context.slots.get("turn.input")
        if isinstance(turn, TurnInput):
            channel, _, chat_id = turn.session_key.partition(":")
            payload.update(
                {
                    "session_key": turn.session_key,
                    "channel": channel,
                    "chat_id": chat_id,
                    "content": turn.content,
                    "source": turn.prompt_scope,
                }
            )
        if self._iteration is not None:
            payload["iteration"] = self._iteration
        response = self._context.slots.get("step.response")
        if isinstance(response, ModelResponse):
            payload["response"] = response
        reasoning_result = self._context.slots.get("reasoning.result")
        if isinstance(reasoning_result, ReActResult):
            payload["reply"] = reasoning_result.reply
            payload["tools_used"] = tuple(
                record.call.name for record in reasoning_result.tool_chain
            )
        return payload

    def _current_phase(self) -> LifecyclePhase:
        phase = self._context.slots.get("runtime.current_phase")
        if not isinstance(phase, LifecyclePhase):
            raise RuntimeError("当前生命周期 Phase 未设置")
        return phase

    def _prepare_phase_contract(self, phase: LifecyclePhase) -> None:
        self._context.slots["runtime.current_phase"] = phase
        state = self._context.slots.get("before_turn.input")
        if not isinstance(state, BeforeTurnInput):
            raise RuntimeError("BeforeTurnInput 未初始化")
        phase_input: Any
        if phase is LifecyclePhase.BEFORE_TURN:
            self._context.slots["session:ctx"] = BeforeTurnCtx(
                session_key=state.session_key,
                channel=state.channel,
                chat_id=state.chat_id,
                content=state.content,
                system_prompt=state.system_prompt,
                history=state.history,
                prompt_scope=state.prompt_scope,
                media=state.media,
                outbound_metadata=dict(state.outbound_metadata),
            )
        elif phase is LifecyclePhase.BEFORE_REASONING:
            before_turn = cast(BeforeTurnCtx, self._context.slots["session:ctx"])
            phase_input = BeforeReasoningInput(state, before_turn)
            self._context.slots["before_reasoning.input"] = phase_input
            self._context.slots["reasoning:ctx"] = BeforeReasoningCtx(
                session_key=before_turn.session_key,
                channel=before_turn.channel,
                chat_id=before_turn.chat_id,
                content=before_turn.content,
                media=before_turn.media,
                history=before_turn.history,
                prompt_scope=before_turn.prompt_scope,
                system_prompt=before_turn.system_prompt,
                extra_hints=list(before_turn.extra_hints),
                visible_tool_names=frozenset(self._tools.tool_names),
            )
        elif phase is LifecyclePhase.PROMPT_RENDER:
            reasoning = cast(BeforeReasoningCtx, self._context.slots["reasoning:ctx"])
            phase_input = PromptRenderInput(
                reasoning.session_key,
                reasoning.channel,
                reasoning.chat_id,
                reasoning.content,
                reasoning.media,
                reasoning.history,
                reasoning.prompt_scope,
                reasoning.system_prompt,
                tuple(reasoning.extra_hints),
            )
            self._context.slots["prompt_render.input"] = phase_input
            self._context.slots["prompt:ctx"] = PromptRenderContext(
                session_key=phase_input.session_key,
                content=phase_input.content,
                scope=phase_input.prompt_scope,
                system_prompt=phase_input.system_prompt,
                history=phase_input.history,
                channel=phase_input.channel,
                chat_id=phase_input.chat_id,
                media=phase_input.media,
                extra_hints=list(phase_input.extra_hints),
            )
        elif phase is LifecyclePhase.BEFORE_STEP:
            messages = cast(tuple[ChatMessage, ...], self._context.slots["step.messages"])
            iteration = cast(int, self._context.slots["step.iteration"])
            visible_names = cast(
                frozenset[str] | None,
                self._context.slots.get("runtime.visible_tool_names"),
            )
            phase_input = BeforeStepInput(
                state.session_key,
                state.channel,
                state.chat_id,
                iteration,
                messages,
                visible_names,
            )
            self._context.slots["before_step.input"] = phase_input
            self._context.slots["step:ctx"] = BeforeStepCtx(
                state.session_key,
                state.channel,
                state.chat_id,
                iteration,
                _estimate_messages_chars(messages),
                phase_input.visible_names,
            )
        elif phase is LifecyclePhase.AFTER_STEP:
            response = cast(ModelResponse, self._context.slots["step.response"])
            records = cast(tuple[ToolCallRecord, ...], self._context.slots.get("step.records", ()))
            messages = cast(tuple[ChatMessage, ...], self._context.slots.get("step.messages", ()))
            step_ctx = AfterStepCtx(
                state.session_key,
                state.channel,
                state.chat_id,
                cast(int, self._context.slots["step.iteration"]),
                _estimate_messages_chars(messages),
                tuple(record.call.name for record in records),
                response.content or "",
                tuple(record.call.name for record in records),
                tuple(_tool_record_snapshot(record) for record in records),
                response.thinking,
                bool(response.tool_calls),
            )
            self._context.slots["after_step.input"] = step_ctx
            self._context.slots["step:ctx"] = step_ctx
        elif phase is LifecyclePhase.AFTER_REASONING:
            phase_input = cast(AfterReasoningInput, self._context.slots["after_reasoning.input"])
            result = cast(ReActResult, phase_input.turn_result)
            self._context.slots["reasoning:ctx"] = AfterReasoningCtx(
                state.session_key,
                state.channel,
                state.chat_id,
                tuple(record.call.name for record in result.tool_chain),
                result.thinking,
                tuple(_tool_record_snapshot(record) for record in result.tool_chain),
                result.reply,
                outbound_metadata=dict(state.outbound_metadata),
            )
        elif phase is LifecyclePhase.AFTER_TURN:
            reasoning_ctx = cast(AfterReasoningCtx, self._context.slots["reasoning:ctx"])
            output = cast(ReActResult, self._context.slots["turn.output"])
            after_result = AfterReasoningResult(reasoning_ctx, output)
            phase_input = AfterTurnInput(state, after_result)
            self._context.slots["after_turn.input"] = phase_input
            self._context.slots["turn:ctx"] = AfterTurnCtx(
                state.session_key,
                state.channel,
                state.chat_id,
                reasoning_ctx.reply,
                reasoning_ctx.tools_used,
                reasoning_ctx.thinking,
                tuple(reasoning_ctx.media),
                dict(reasoning_ctx.outbound_metadata),
            )

    async def before_step(
        self,
        iteration: int,
        messages: Sequence[ChatMessage],
    ) -> BeforeStepControl:
        self._iteration = iteration
        self._context.slots["step.iteration"] = iteration
        self._context.slots["step.messages"] = tuple(messages)
        await self.run_phase(LifecyclePhase.BEFORE_STEP)
        ctx = cast(BeforeStepCtx, self._context.slots["step:ctx"])
        return BeforeStepControl(
            tuple(ctx.extra_hints),
            ctx.early_stop,
            ctx.early_stop_reply,
        )

    def set_visible_tool_names(self, names: frozenset[str]) -> None:
        self._context.slots["runtime.visible_tool_names"] = names

    async def after_step(
        self,
        iteration: int,
        response: ModelResponse,
        tool_records: Sequence[ToolCallRecord],
    ) -> AfterStepControl:
        self._iteration = iteration
        self._context.slots["step.response"] = response
        self._context.slots["step.records"] = tuple(tool_records)
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
                    state=(
                        "succeeded"
                        if record.observation.status == "success"
                        else record.observation.status
                    ),
                    iteration=iteration,
                    tool_name=record.call.name,
                    input=record.call.arguments,
                    observation={
                        "ok": record.observation.ok,
                        "status": record.observation.status,
                        "result": record.observation.result,
                        "error_type": record.observation.error_type,
                        "error_message": record.observation.error_message,
                        "original_arguments": record.observation.original_arguments,
                        "final_arguments": record.observation.final_arguments,
                        "hook_trace": [
                            asdict(item) for item in record.observation.hook_trace
                        ],
                        "extra_messages": record.observation.extra_messages,
                        "retryable": record.observation.retryable,
                    },
                )
            )
        await self.run_phase(LifecyclePhase.AFTER_STEP)
        ctx = cast(AfterStepCtx, self._context.slots["step:ctx"])
        return AfterStepControl(ctx.early_stop, ctx.early_stop_reason)

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
    return {"session:ctx": context.slots["session:ctx"]}


async def _emit_before_turn(
    context: PhaseContext,
    *,
    event_bus: EventBus | None,
) -> Mapping[str, Any]:
    await _emit_gate(context, LifecyclePhase.BEFORE_TURN, event_bus, "session:ctx")
    return {}


async def _finalize_before_turn(context: PhaseContext) -> Mapping[str, Any]:
    turn = cast(TurnInput, context.slots["turn.input"])
    ctx = cast(BeforeTurnCtx, context.slots["session:ctx"])
    state = cast(BeforeTurnInput, context.slots["before_turn.input"])
    content = state.content if state.content != turn.content else ctx.content
    system_prompt = (
        state.system_prompt if state.system_prompt != turn.system_prompt else ctx.system_prompt
    )
    effective = replace(
        turn,
        content=content,
        system_prompt=system_prompt,
        history=ctx.history,
        media=ctx.media,
        outbound_metadata=dict(ctx.outbound_metadata),
    )
    context.slots["turn.input"] = effective
    context.slots["before_turn.input"] = _turn_state(effective)
    return {"turn.request": effective}


async def _before_reasoning(context: PhaseContext) -> Mapping[str, Any]:
    return {"reasoning:ctx": context.slots["reasoning:ctx"]}


async def _emit_before_reasoning(
    context: PhaseContext,
    *,
    event_bus: EventBus | None,
) -> Mapping[str, Any]:
    await _emit_gate(
        context,
        LifecyclePhase.BEFORE_REASONING,
        event_bus,
        "reasoning:ctx",
    )
    return {}


async def _finalize_before_reasoning(context: PhaseContext) -> Mapping[str, Any]:
    turn = cast(TurnInput, context.slots["turn.request"])
    ctx = cast(BeforeReasoningCtx, context.slots["reasoning:ctx"])
    effective = replace(
        turn,
        content=ctx.content,
        system_prompt=ctx.system_prompt,
        history=ctx.history,
        media=ctx.media,
    )
    return {"reasoning.input": effective}


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
    prompt_context = context.slots.get("prompt:ctx")
    if not isinstance(prompt_context, PromptRenderContext):
        raise TypeError("prompt:ctx 必须是 PromptRenderContext")
    messages: list[ChatMessage] = []
    core_prompt = turn.system_prompt.strip()
    if prompt_context.extra_hints:
        core_prompt = "\n\n".join(
            part
            for part in (
                core_prompt,
                "运行时补充提示：\n" + "\n".join(prompt_context.extra_hints),
            )
            if part
        )
    memory_prompt = ""
    memory_context = context.slots.get("memory.context")
    if isinstance(memory_context, str) and memory_context.strip():
        memory_prompt = (
            "以下是与当前问题相关的长期记忆：\n"
            + memory_context.strip()
            + "\n\n"
            + CITATION_PROTOCOL
        )
    active_prompt = ""
    skill_catalog = ""
    if skills is not None:
        skill_prompt = skills.build_turn_prompt(turn.content)
        skill_catalog = skill_prompt.catalog
        active_prompt = _allocate_active_skills(
            skill_prompt.active,
            max_chars=max_chars,
            reserved_core_chars=len(core_prompt),
        )
    active_separator = 2 if active_prompt else 0
    non_skill_budget = max_chars - len(active_prompt) - active_separator
    system_prompt = renderer.render(
        scope=turn.prompt_scope,
        base_prompt="\n\n".join(part for part in (core_prompt, memory_prompt) if part),
        max_chars=non_skill_budget,
        top_sections=tuple(prompt_context.system_sections_top),
        bottom_sections=tuple(prompt_context.system_sections_bottom),
    )
    system_prompt = _append_prompt_text(system_prompt, active_prompt, max_chars=max_chars)
    system_prompt = _append_prompt_text(system_prompt, skill_catalog, max_chars=max_chars)
    if system_prompt:
        messages.append(ChatMessage.system(system_prompt))
    messages.extend(turn.history)
    messages.append(ChatMessage.user(turn.content))
    rendered = tuple(messages)
    context.slots["prompt_render.output"] = PromptRenderResult(rendered)
    return {"prompt.messages": rendered}


def _allocate_active_skills(
    blocks: tuple[ActiveSkillBlock, ...],
    *,
    max_chars: int,
    reserved_core_chars: int,
) -> str:
    """核心 Prompt 先保底；Skill 正文只能整块进入，目录不参与本次分配。"""
    available = max_chars - reserved_core_chars - (2 if reserved_core_chars else 0)
    rendered: list[str] = []
    omitted: list[str] = []
    used = 0
    for block in blocks:
        separator = 2 if rendered else 0
        if used + separator + len(block.content) <= available:
            rendered.append(block.content)
            used += separator + len(block.content)
        else:
            omitted.append(block.name)

    if omitted:
        diagnostic = (
            "# Skill 注入诊断\n"
            "Skill 正文因超出 Prompt 预算未注入: " + ", ".join(omitted)
        )
        while rendered and used + 2 + len(diagnostic) > available:
            removed = rendered.pop()
            used -= len(removed) + (2 if rendered else 0)
            name = removed.partition("\n")[0].removeprefix("# Skill: ")
            omitted.append(name)
            diagnostic = (
                "# Skill 注入诊断\n"
                "Skill 正文因超出 Prompt 预算未注入: "
                + ", ".join(sorted(omitted))
            )
        separator = 2 if rendered else 0
        if used + separator + len(diagnostic) <= available:
            rendered.append(diagnostic)
    return "\n\n".join(rendered)


def _append_prompt_text(current: str, content: str, *, max_chars: int) -> str:
    text = content.strip()
    if not text or len(current) >= max_chars:
        return current
    separator = "\n\n" if current else ""
    remaining = max_chars - len(current) - len(separator)
    if remaining <= 0:
        return current
    return current + separator + text[:remaining]


async def _build_prompt_context(context: PhaseContext) -> Mapping[str, Any]:
    turn = context.slots["reasoning.input"]
    if not isinstance(turn, TurnInput):
        raise TypeError("reasoning.input 必须是 TurnInput")
    existing = context.slots.get("prompt:ctx")
    if isinstance(existing, PromptRenderContext):
        return {"prompt:ctx": existing}
    return {
        "prompt:ctx": PromptRenderContext(
            session_key=turn.session_key,
            content=turn.content,
            scope=turn.prompt_scope,
            system_prompt=turn.system_prompt,
            history=turn.history,
            channel=turn.session_key.partition(":")[0],
            chat_id=turn.session_key.partition(":")[2],
            media=turn.media,
            extra_hints=list(
                cast(BeforeReasoningCtx, context.slots["reasoning:ctx"]).extra_hints
            ),
        )
    }


async def _emit_prompt_context(
    context: PhaseContext,
    *,
    event_bus: EventBus | None,
) -> Mapping[str, Any]:
    if not isinstance(context.slots.get("prompt:ctx"), PromptRenderContext):
        raise TypeError("prompt:ctx 必须是 PromptRenderContext")
    await _emit_gate(context, LifecyclePhase.PROMPT_RENDER, event_bus, "prompt:ctx")
    return {}


async def _before_step(context: PhaseContext) -> Mapping[str, Any]:
    return {"step:ctx": context.slots["step:ctx"]}


async def _emit_before_step(
    context: PhaseContext,
    *,
    event_bus: EventBus | None,
) -> Mapping[str, Any]:
    await _emit_gate(context, LifecyclePhase.BEFORE_STEP, event_bus, "step:ctx")
    return {}


async def _finalize_before_step(context: PhaseContext) -> Mapping[str, Any]:
    ctx = cast(BeforeStepCtx, context.slots["step:ctx"])
    return {
        "step.control": BeforeStepControl(
            tuple(ctx.extra_hints),
            ctx.early_stop,
            ctx.early_stop_reply,
        )
    }


async def _after_step(context: PhaseContext) -> Mapping[str, Any]:
    return {"step:ctx": context.slots["step:ctx"]}


async def _observe_after_step(context: PhaseContext) -> Mapping[str, Any]:
    return {"step.observed": context.slots["step:ctx"]}


async def _after_reasoning(context: PhaseContext) -> Mapping[str, Any]:
    return {"reasoning:ctx": context.slots["reasoning:ctx"]}


async def _emit_after_reasoning(
    context: PhaseContext,
    *,
    event_bus: EventBus | None,
) -> Mapping[str, Any]:
    await _emit_gate(
        context,
        LifecyclePhase.AFTER_REASONING,
        event_bus,
        "reasoning:ctx",
    )
    return {}


async def _finalize_after_reasoning(context: PhaseContext) -> Mapping[str, Any]:
    result = cast(ReActResult, context.slots["reasoning.result"])
    ctx = cast(AfterReasoningCtx, context.slots["reasoning:ctx"])
    updated = _replace_react_reply(result, ctx.reply)
    context.slots["reasoning.result"] = updated
    context.slots["after_reasoning.output"] = AfterReasoningResult(ctx, updated)
    return {"turn.output": updated}


async def _after_turn(context: PhaseContext) -> Mapping[str, Any]:
    return {"turn:ctx": context.slots["turn:ctx"]}


async def _complete_after_turn(context: PhaseContext) -> Mapping[str, Any]:
    return {"turn.completed": context.slots["turn.output"] is not None}


def _default_modules(
    prompt_renderer: PromptRenderer,
    prompt_max_chars: int,
    skills: SkillCatalog | None,
    event_bus: EventBus | None,
) -> tuple[FunctionPhaseModule, ...]:
    return (
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_TURN,
            "before_turn.build_ctx",
            ("turn.input",),
            ("session:ctx",),
            _before_turn,
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_TURN,
            "before_turn.emit",
            ("before_turn.build_ctx", "session:ctx"),
            (),
            partial(_emit_before_turn, event_bus=event_bus),
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_TURN,
            "before_turn.finalize",
            ("before_turn.emit", "session:ctx"),
            ("turn.request",),
            _finalize_before_turn,
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_REASONING,
            "before_reasoning.build_ctx",
            ("turn.request",),
            ("reasoning:ctx",),
            _before_reasoning,
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_REASONING,
            "before_reasoning.emit",
            ("before_reasoning.build_ctx", "reasoning:ctx"),
            (),
            partial(_emit_before_reasoning, event_bus=event_bus),
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_REASONING,
            "before_reasoning.finalize",
            ("before_reasoning.emit", "reasoning:ctx"),
            ("reasoning.input",),
            _finalize_before_reasoning,
        ),
        FunctionPhaseModule(
            LifecyclePhase.PROMPT_RENDER,
            "prompt_render.build_ctx",
            ("reasoning.input",),
            ("prompt:ctx",),
            _build_prompt_context,
        ),
        FunctionPhaseModule(
            LifecyclePhase.PROMPT_RENDER,
            "prompt_render.emit",
            ("prompt_render.build_ctx", "prompt:ctx"),
            (),
            partial(_emit_prompt_context, event_bus=event_bus),
        ),
        FunctionPhaseModule(
            LifecyclePhase.PROMPT_RENDER,
            "prompt_render.messages",
            ("prompt_render.emit", "prompt:ctx"),
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
            "before_step.build_ctx",
            ("prompt.messages", "step.iteration", "step.messages"),
            ("step:ctx",),
            _before_step,
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_STEP,
            "before_step.emit",
            ("before_step.build_ctx", "step:ctx"),
            (),
            partial(_emit_before_step, event_bus=event_bus),
        ),
        FunctionPhaseModule(
            LifecyclePhase.BEFORE_STEP,
            "before_step.finalize",
            ("before_step.emit", "step:ctx"),
            ("step.control",),
            _finalize_before_step,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_STEP,
            "after_step.copy_input",
            ("step.response",),
            ("step:ctx",),
            _after_step,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_STEP,
            "after_step.observe",
            ("after_step.copy_input", "step:ctx"),
            ("step.observed",),
            _observe_after_step,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_REASONING,
            "after_reasoning.build_ctx",
            ("reasoning.result",),
            ("reasoning:ctx",),
            _after_reasoning,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_REASONING,
            "after_reasoning.emit",
            ("after_reasoning.build_ctx", "reasoning:ctx"),
            (),
            partial(_emit_after_reasoning, event_bus=event_bus),
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_REASONING,
            "after_reasoning.finalize",
            ("after_reasoning.emit", "reasoning:ctx"),
            ("turn.output",),
            _finalize_after_reasoning,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_TURN,
            "after_turn.build_ctx",
            ("turn.output",),
            ("turn:ctx",),
            _after_turn,
        ),
        FunctionPhaseModule(
            LifecyclePhase.AFTER_TURN,
            "after_turn.complete",
            ("after_turn.build_ctx", "turn:ctx"),
            ("turn.completed",),
            _complete_after_turn,
        ),
    )


def validate_extension_phase_modules(modules: Sequence[PhaseModule]) -> None:
    """用真实核心 Phase 合同在插件提交前验证扩展模块。"""
    all_modules = cast(
        Sequence[PhaseModule],
        (*_default_modules(PromptRenderer(), 12000, None, None), *modules),
    )
    PhasePipeline(
        all_modules,
        initial_slots={"turn.input"},
        provided_slots={
            LifecyclePhase.BEFORE_REASONING: {"memory.context"},
            LifecyclePhase.BEFORE_STEP: {"step.iteration", "step.messages"},
            LifecyclePhase.AFTER_STEP: {"step.response"},
            LifecyclePhase.AFTER_REASONING: {"reasoning.result"},
        },
    )


def _turn_state(turn: TurnInput) -> BeforeTurnInput:
    channel, _, chat_id = turn.session_key.partition(":")
    return BeforeTurnInput(
        session_key=turn.session_key,
        channel=channel,
        chat_id=chat_id,
        content=turn.content,
        system_prompt=turn.system_prompt,
        history=turn.history,
        prompt_scope=turn.prompt_scope,
        media=turn.media,
        outbound_metadata=dict(turn.outbound_metadata),
    )


async def _emit_gate(
    context: PhaseContext,
    phase: LifecyclePhase,
    event_bus: EventBus | None,
    ctx_slot: str,
) -> None:
    if event_bus is None:
        return
    ctx = context.slots[ctx_slot]
    payload = dict(context.slots)
    payload.update(asdict(ctx))
    updated = await event_bus.emit(phase.value, payload)
    for key, value in updated.items():
        if hasattr(ctx, key):
            setattr(ctx, key, value)


def _phase_ctx_slot(phase: LifecyclePhase) -> str:
    return {
        LifecyclePhase.BEFORE_TURN: "session:ctx",
        LifecyclePhase.BEFORE_REASONING: "reasoning:ctx",
        LifecyclePhase.PROMPT_RENDER: "prompt:ctx",
        LifecyclePhase.BEFORE_STEP: "step:ctx",
        LifecyclePhase.AFTER_STEP: "step:ctx",
        LifecyclePhase.AFTER_REASONING: "reasoning:ctx",
        LifecyclePhase.AFTER_TURN: "turn:ctx",
    }[phase]


def _estimate_messages_chars(messages: Sequence[ChatMessage]) -> int:
    return sum(len(item.content or "") for item in messages)


def _tool_record_snapshot(record: ToolCallRecord) -> dict[str, object]:
    return {
        "iteration": record.iteration,
        "name": record.call.name,
        "arguments": dict(record.call.arguments),
        "status": record.observation.status,
        "result": record.observation.result,
    }


def _replace_react_reply(result: ReActResult, reply: str) -> ReActResult:
    messages = list(result.messages)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            messages[index] = replace(messages[index], content=reply)
            break
    else:
        messages.append(ChatMessage.assistant(content=reply))
    return replace(result, reply=reply, messages=tuple(messages))


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
    "validate_extension_phase_modules",
]
