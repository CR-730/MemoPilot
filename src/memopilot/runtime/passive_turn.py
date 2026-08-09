"""Passive reply persistence and delivery pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from memopilot.bus.events import InboundMessage, TurnCommitted
from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_events import (
    AfterReasoningCtx,
    AfterReasoningInput,
    AfterReasoningResult,
    AfterTurnCtx,
    AfterTurnInput,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeTurnCtx,
    BeforeTurnInput,
)
from memopilot.persistence.conversation import ConversationRepository, TurnCommitResult
from memopilot.runtime.engine import (
    DefaultReasoner,
    FunctionPhaseModule,
    PhaseTraceEntry,
    ReasoningSession,
    TurnInput,
    TurnResult,
    build_outer_modules,
)
from memopilot.runtime.history import build_tool_chain
from memopilot.runtime.outbound import DeliveryError, OutboundDispatch, OutboundPort
from memopilot.runtime.phases import LifecyclePhase, PhaseContext, PhaseModule, PhasePipeline
from memopilot.runtime.react import ReActProgressObserver, ReActResult, ToolCallRecord
from memopilot.runtime.session import OperationalSessionManager
from memopilot.runtime.tools import ToolRegistry
from memopilot.tasks.agent_task import AgentTask


class PassiveTurnPipeline:
    def __init__(
        self,
        runtime: DefaultReasoner,
        *,
        repository: ConversationRepository,
        outbound: OutboundPort,
        event_bus: EventBus,
        history_limit: int,
        progress_factory: Callable[[InboundMessage], ReActProgressObserver | None] | None = None,
        after_turn_modules: Sequence[PhaseModule] = (),
        outer_modules: Sequence[PhaseModule] = (),
    ) -> None:
        self._runtime, self._repository, self._outbound = runtime, repository, outbound
        self._event_bus, self._history_limit, self._progress_factory = (
            event_bus,
            history_limit,
            progress_factory,
        )
        self._outer = PhasePipeline(
            cast(Sequence[PhaseModule], (*build_outer_modules(
                self._event_bus,
                memory_modules=cast(Sequence[PhaseModule], (
                    FunctionPhaseModule(
                        LifecyclePhase.BEFORE_REASONING,
                        "before_reasoning.memory_prerecall",
                        ("reasoning.input",),
                        ("memory.context", "memory.trace"),
                        self._runtime._memory_prerecall,
                    ),
                )) if (
                    self._runtime._memory_engine is not None
                    or self._runtime._memory_profile is not None
                ) else (),
                plugin_modules=outer_modules,
            ), FunctionPhaseModule(
                LifecyclePhase.AFTER_REASONING,
                "after_reasoning.persist_turn",
                ("after_reasoning.finalize", "turn.output", "passive.message"),
                ("passive.committed",),
                self._persist_after_reasoning,
            ))),
            initial_slots={
                "turn.input", "before_turn.input", "passive.message",
                "passive.direct", "passive.source_ref",
            },
            provided_slots={LifecyclePhase.AFTER_REASONING: {"reasoning.result"}},
        )
        gated_after_turn_modules = tuple(
            _AfterTurnPluginGate(module) for module in after_turn_modules
        )
        self._after_turn = PhasePipeline(
            cast(
                Sequence[PhaseModule],
                (
                    FunctionPhaseModule(
                        LifecyclePhase.AFTER_TURN,
                        "after_turn.build_ctx",
                        ("after_turn.input",),
                        ("turn:ctx",),
                        _after_turn_ctx,
                    ),
                    FunctionPhaseModule(
                        LifecyclePhase.AFTER_TURN,
                        "after_turn.emit_committed",
                        ("after_turn.build_ctx", "turn:ctx"),
                        (),
                        self._emit_committed,
                    ),
                    *gated_after_turn_modules,
                    FunctionPhaseModule(
                        LifecyclePhase.AFTER_TURN,
                        "after_turn.dispatch",
                        (
                            "after_turn.emit_committed",
                            "turn:ctx",
                            *(module.slot for module in gated_after_turn_modules),
                        ),
                        ("turn.dispatch",),
                        self._dispatch,
                    ),
                ),
            ),
            initial_slots={"after_turn.input"},
        )

    async def run(
        self,
        message: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> Sequence[AgentTask]:
        if message.session_key != key:
            raise ValueError("passive turn session_key mismatch")
        session = self._runtime.begin_turn(
            TurnInput(
                key,
                message.content,
                history=OperationalSessionManager(self._repository).get_history(
                    key, self._history_limit
                ),
                current_user_content=message.content,
                media=tuple(message.media),
                received_at=message.timestamp,
                memory_source_ref=self._repository.predict_user_message_id(message),
            )
        )
        session.context.slots.update({
            "passive.message": message,
            "passive.direct": False,
            "passive.source_ref": "",
        })
        await self._run_outer_phase(session, LifecyclePhase.BEFORE_TURN)
        before_turn = cast(BeforeTurnCtx, session.context.slots["session:ctx"])
        if before_turn.abort:
            await self._dispatch_control(message, before_turn.abort_reply, dispatch_outbound)
            return ()
        await self._run_outer_phase(session, LifecyclePhase.BEFORE_REASONING)
        before_reasoning = cast(BeforeReasoningCtx, session.context.slots["reasoning:ctx"])
        if before_reasoning.abort:
            await self._dispatch_control(message, before_reasoning.abort_reply, dispatch_outbound)
            return ()
        progress = self._progress_factory(message) if self._progress_factory else None
        try:
            await self._runtime.run_reasoning(session, progress=progress)
        finally:
            if progress is not None:
                await progress.finalize()  # type: ignore[attr-defined]
        self._runtime.prepare_after_reasoning(session)
        await self._run_outer_phase(session, LifecyclePhase.AFTER_REASONING)
        result = self._runtime.finish_turn(session)
        committed = cast(TurnCommitResult, session.context.slots["passive.committed"])
        tools_used = tuple(dict.fromkeys(item.call.name for item in result.react.tool_chain))
        tool_chain = tuple(_tool_chain_item(item) for item in result.react.tool_chain)
        provider_uuid = str(
            uuid5(
                NAMESPACE_URL,
                "memopilot:reply:" + str(message.metadata.get("message_id") or message.session_key),
            )
        )
        after = await self._run_after_turn(
            message,
            committed=committed,
            tools_used=tools_used,
            tool_chain=tool_chain,
            cache_prompt_tokens=result.react.prompt_tokens,
            cache_hit_tokens=result.react.prompt_cache_hit_tokens,
            thinking=result.react.thinking,
            outbound_metadata=result.outbound_metadata,
            provider_uuid=provider_uuid,
            turn_result=result.react,
            dispatch_outbound=dispatch_outbound,
        )
        if dispatch_outbound and not after.slots["turn.dispatch"]:
            raise DeliveryError("passive reply delivery was not confirmed")
        return _memory_tasks(
            committed.turn_id,
            committed.assistant_message_id,
            message,
            cited_memory_ids=result.cited_memory_ids,
            explicitly_memorized_ids=_explicitly_memorized_ids(result.trace),
        )

    async def _dispatch_control(
        self, message: InboundMessage, reply: str, dispatch_outbound: bool
    ) -> None:
        if dispatch_outbound:
            await self._outbound.dispatch(
                OutboundDispatch(message.channel, message.chat_id, reply)
            )


    async def execute_direct(
        self,
        turn: TurnInput,
        *,
        tools: ToolRegistry | None = None,
        execution_assert_current: Callable[[], None] | None = None,
        memory_assert_current: Callable[[], None] | None = None,
        memory_fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> TurnResult:
        """复用收尾插件链；direct 不写入伪用户消息、不出站也不派生记忆任务。"""
        channel, _, chat_id = turn.session_key.partition(":")
        timestamp = turn.received_at or datetime.now(UTC)
        existing = self._repository.find_direct_assistant(
            session_key=turn.session_key, timestamp=timestamp, source_ref=turn.memory_source_ref
        )
        if existing is not None:
            react = ReActResult(existing.assistant_content, (), 0, (), "replay")
            replay = TurnResult(existing.assistant_content, (), react, (), (), media=existing.media)
            await self._run_after_turn(
                InboundMessage(channel, channel, chat_id, turn.content, timestamp=timestamp),
                committed=existing,
                tools_used=(),
                tool_chain=(),
                cache_prompt_tokens=0,
                cache_hit_tokens=0,
                thinking=None,
                outbound_metadata={},
                provider_uuid="",
                turn_result=react,
                dispatch_outbound=False,
            )
            return replay
        message = InboundMessage(
            channel,
            channel,
            chat_id,
            turn.content,
            timestamp=timestamp,
        )
        session = self._runtime.begin_turn(turn, tools=tools)
        session.context.slots.update({
            "passive.message": message,
            "passive.direct": True,
            "passive.source_ref": turn.memory_source_ref,
        })
        await self._run_outer_phase(session, LifecyclePhase.BEFORE_TURN)
        before_turn = cast(BeforeTurnCtx, session.context.slots["session:ctx"])
        if before_turn.abort:
            return self._runtime.aborted_result(session, before_turn, "before_turn_abort")
        await self._run_outer_phase(session, LifecyclePhase.BEFORE_REASONING)
        before_reasoning = cast(BeforeReasoningCtx, session.context.slots["reasoning:ctx"])
        if before_reasoning.abort:
            return self._runtime.aborted_result(
                session, before_reasoning, "before_reasoning_abort"
            )
        await self._runtime.run_reasoning(
            session,
            execution_assert_current=execution_assert_current,
            memory_assert_current=memory_assert_current,
            memory_fenced_write=memory_fenced_write,
        )
        self._runtime.prepare_after_reasoning(session)
        await self._run_outer_phase(session, LifecyclePhase.AFTER_REASONING)
        result = self._runtime.finish_turn(session)
        committed = cast(TurnCommitResult, session.context.slots["passive.committed"])
        await self._run_after_turn(
            message,
            committed=committed,
            tools_used=tuple(dict.fromkeys(item.call.name for item in result.react.tool_chain)),
            tool_chain=tuple(_tool_chain_item(item) for item in result.react.tool_chain),
            cache_prompt_tokens=result.react.prompt_tokens,
            cache_hit_tokens=result.react.prompt_cache_hit_tokens,
            thinking=result.react.thinking,
            outbound_metadata=result.outbound_metadata,
            provider_uuid="",
            turn_result=result.react,
            dispatch_outbound=False,
        )
        return result

    async def _run_outer_phase(
        self, session: ReasoningSession, phase: LifecyclePhase
    ) -> None:
        _prepare_outer_phase(session.context, phase, session.tools)
        trace_start = len(session.context.trace)
        await self._outer.run_phase(phase, session.context)
        session.execution.phase_trace.extend(
            PhaseTraceEntry(traced_phase, slot, None)
            for traced_phase, slot in session.context.trace[trace_start:]
        )

    async def _run_after_turn(
        self,
        message: InboundMessage,
        *,
        committed: TurnCommitResult,
        tools_used: tuple[str, ...],
        tool_chain: tuple[dict[str, object], ...],
        cache_prompt_tokens: int,
        cache_hit_tokens: int,
        thinking: str | None,
        outbound_metadata: Mapping[str, object],
        provider_uuid: str,
        turn_result: ReActResult,
        dispatch_outbound: bool = True,
    ) -> PhaseContext:
        context = PhaseContext(
            slots={
                "after_turn.input": AfterTurnInput(
                    _after_turn_state(message),
                    AfterReasoningResult(
                        AfterReasoningCtx(
                            message.session_key,
                            message.channel,
                            message.chat_id,
                            tuple(tools_used),
                            thinking,
                            tool_chain,
                            committed.assistant_content,
                            list(committed.media),
                            dict(outbound_metadata),
                        ),
                        turn_result,
                    ),
                ),
                "after_turn.committed": committed,
                "after_turn.event": TurnCommitted(
                    message.session_key,
                    message.channel,
                    message.chat_id,
                    message.content,
                    committed.assistant_content,
                    list(tools_used),
                    message.timestamp,
                    cast(tuple[dict[str, str], ...], tool_chain),
                    cache_prompt_tokens,
                    cache_hit_tokens,
                ),
                "after_turn.dispatch_outbound": dispatch_outbound,
                "after_turn.provider_uuid": provider_uuid,
                "after_turn.thinking": thinking,
            }
        )
        await self._after_turn.run_phase(LifecyclePhase.AFTER_TURN, context)
        return context

    async def _emit_committed(self, context: PhaseContext) -> dict[str, object]:
        committed = context.slots["after_turn.committed"]
        if committed.inserted:
            await self._event_bus.fanout(context.slots["after_turn.event"])
        return {}

    async def _persist_after_reasoning(self, context: PhaseContext) -> dict[str, object]:
        message = cast(InboundMessage, context.slots["passive.message"])
        result = cast(ReActResult, context.slots["turn.output"])
        reasoning = cast(AfterReasoningCtx, context.slots["reasoning:ctx"])
        if context.slots["passive.direct"]:
            committed = self._repository.commit_direct_assistant(
                session_key=message.session_key,
                channel=message.channel,
                chat_id=message.chat_id,
                content=result.reply,
                media=tuple(reasoning.media),
                timestamp=message.timestamp,
                source_ref=cast(str, context.slots["passive.source_ref"]),
            )
        else:
            committed = self._repository.commit_turn(
                message,
                assistant_content=result.reply,
                assistant_media=tuple(reasoning.media),
                assistant_tool_chain=build_tool_chain(
                    result.messages,
                    call_ids=frozenset(item.call.id for item in result.tool_chain),
                ),
            )
        return {"passive.committed": committed}

    async def _dispatch(self, context: PhaseContext) -> dict[str, object]:
        if not context.slots["after_turn.dispatch_outbound"]:
            return {"turn.dispatch": True}
        ctx = context.slots["turn:ctx"]
        if not isinstance(ctx, AfterTurnCtx):
            raise TypeError("turn:ctx must be AfterTurnCtx")
        return {
            "turn.dispatch": await self._outbound.dispatch(
                OutboundDispatch(
                    ctx.channel,
                    ctx.chat_id,
                    ctx.reply,
                    thinking=ctx.thinking,
                    metadata={
                        **ctx.outbound_metadata,
                        "provider_uuid": context.slots["after_turn.provider_uuid"],
                    },
                    media=list(ctx.media),
                )
            )
        }


@dataclass(frozen=True)
class _AfterTurnPluginGate:
    delegate: PhaseModule

    phase = LifecyclePhase.AFTER_TURN

    @property
    def slot(self) -> str:
        return self.delegate.slot

    @property
    def requires(self) -> tuple[str, ...]:
        return (*self.delegate.requires, "after_turn.emit_committed")

    @property
    def produces(self) -> tuple[str, ...]:
        return self.delegate.produces

    @property
    def optional_produces(self) -> tuple[str, ...]:
        return tuple(getattr(self.delegate, "optional_produces", ()))

    async def run(self, context: PhaseContext) -> Mapping[str, object]:
        return await self.delegate.run(context)


def _inbound_message(payload: Mapping[str, object]) -> InboundMessage:
    timestamp = datetime.fromisoformat(str(payload.get("timestamp") or ""))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    metadata, media = payload.get("metadata"), payload.get("media")
    channel, sender, chat_id, content = (
        str(payload.get(key) or "").strip() for key in ("channel", "sender", "chat_id", "content")
    )
    if not all((channel, sender, chat_id, content)):
        raise ValueError("passive.turn 缺少 channel、sender、chat_id 或 content")
    return InboundMessage(
        channel,
        sender,
        chat_id,
        content,
        timestamp=timestamp,
        media=[str(item) for item in media] if isinstance(media, list) else [],
        metadata={str(key): value for key, value in metadata.items()}
        if isinstance(metadata, Mapping)
        else {},
    )


def _after_turn_state(message: InboundMessage) -> BeforeTurnInput:
    return BeforeTurnInput(
        message.session_key,
        message.channel,
        message.chat_id,
        message.content,
        "",
        (),
        "passive",
        tuple(message.media),
    )


def _prepare_outer_phase(
    context: PhaseContext,
    phase: LifecyclePhase,
    tools: ToolRegistry,
) -> None:
    """Prepare only the outer lifecycle contracts owned by this pipeline."""
    state = context.slots.get("before_turn.input")
    if not isinstance(state, BeforeTurnInput):
        raise RuntimeError("BeforeTurnInput 未初始化")
    if phase is LifecyclePhase.BEFORE_TURN:
        context.slots["session:ctx"] = BeforeTurnCtx(
            state.session_key, state.channel, state.chat_id, state.content,
            state.system_prompt, state.history, state.prompt_scope, state.media,
            outbound_metadata=dict(state.outbound_metadata),
        )
    elif phase is LifecyclePhase.BEFORE_REASONING:
        before_turn = context.slots.get("session:ctx")
        if not isinstance(before_turn, BeforeTurnCtx):
            raise RuntimeError("BeforeTurnCtx 未初始化")
        context.slots["before_reasoning.input"] = BeforeReasoningInput(state, before_turn)
        context.slots["reasoning:ctx"] = BeforeReasoningCtx(
            before_turn.session_key, before_turn.channel, before_turn.chat_id,
            before_turn.content, before_turn.media, before_turn.history,
            before_turn.prompt_scope, before_turn.system_prompt,
            extra_hints=list(before_turn.extra_hints),
            visible_tool_names=frozenset(tools.tool_names),
        )
    elif phase is LifecyclePhase.AFTER_REASONING:
        result = context.slots.get("reasoning.result")
        if not isinstance(result, ReActResult):
            raise RuntimeError("ReActResult 未初始化")
        context.slots["after_reasoning.input"] = AfterReasoningInput(state, result)
        context.slots["reasoning:ctx"] = AfterReasoningCtx(
            state.session_key, state.channel, state.chat_id,
            tuple(record.call.name for record in result.tool_chain), result.thinking,
            tuple(_tool_chain_item(record) for record in result.tool_chain), result.reply,
            outbound_metadata=dict(state.outbound_metadata),
        )


async def _after_turn_ctx(context: PhaseContext) -> dict[str, object]:
    message = context.slots["after_turn.input"].state
    reasoning = context.slots["after_turn.input"].result.ctx
    committed = context.slots["after_turn.committed"]
    return {
        "turn:ctx": AfterTurnCtx(
            message.session_key,
            message.channel,
            message.chat_id,
            committed.assistant_content,
            reasoning.tools_used,
            reasoning.thinking,
            tuple(committed.media),
            dict(context.slots["after_turn.input"].result.ctx.outbound_metadata),
            bool(context.slots["after_turn.dispatch_outbound"]),
        )
    }


def _tool_chain_item(item: ToolCallRecord) -> dict[str, object]:
    call, observation = item.call, item.observation
    return {
        "name": call.name,
        "status": observation.status,
        ("result" if observation.ok else "error"): (
            observation.content
            if observation.ok
            else str(observation.error_message or observation.error_type or "")
        ),
    }


def _explicitly_memorized_ids(trace: Sequence[object]) -> tuple[str, ...]:
    item_ids: list[str] = []
    for event in trace:
        if getattr(event, "tool_name", None) != "memorize":
            continue
        if getattr(event, "state", None) != "succeeded":
            continue
        observation = getattr(event, "observation", None)
        result = observation.get("result") if isinstance(observation, dict) else None
        item_id = result.get("item_id") if isinstance(result, dict) else None
        if isinstance(item_id, str) and item_id.strip():
            item_ids.append(item_id.strip())
    return tuple(dict.fromkeys(item_ids))


def _memory_tasks(
    turn_id: str,
    assistant_message_id: str,
    message: InboundMessage,
    *,
    cited_memory_ids: tuple[str, ...],
    explicitly_memorized_ids: tuple[str, ...],
) -> tuple[AgentTask, ...]:
    def task_id(value: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"memopilot:task:{value}"))

    tasks = [
        AgentTask(
            task_id(f"consolidate:{turn_id}"),
            "memory.consolidate",
            3,
            message.session_key,
            {
                "trigger_turn_id": turn_id,
                "last_message_id": assistant_message_id,
            },
            message.timestamp,
        ),
        AgentTask(
            task_id(f"post-response:{turn_id}"),
            "memory.post_response",
            3,
            message.session_key,
            {
                "turn_id": turn_id,
                "protected_ids": list(
                    dict.fromkeys(item.strip() for item in explicitly_memorized_ids if item.strip())
                ),
            },
            message.timestamp,
        ),
    ]
    cited = tuple(dict.fromkeys(item.strip() for item in cited_memory_ids if item.strip()))
    if cited:
        tasks.append(
            AgentTask(
                task_id(f"memory-reinforce:{turn_id}"),
                "memory.reinforce",
                3,
                message.session_key,
                {"usage_ref": f"turn:{turn_id}", "item_ids": list(cited)},
                message.timestamp,
            )
        )
    return tuple(tasks)


__all__ = ["PassiveTurnPipeline"]
