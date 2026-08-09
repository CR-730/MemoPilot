from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memopilot.bus.events import InboundMessage, TurnCommitted
from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_events import AfterTurnCtx, AfterTurnInput
from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.agent_core import AgentCore
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.engine import DefaultReasoner, TurnInput
from memopilot.runtime.outbound import OutboundDispatch
from memopilot.runtime.passive_turn import PassiveTurnPipeline
from memopilot.runtime.phases import LifecyclePhase, PhaseContext
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.react import ReActResult
from memopilot.runtime.task_dispatcher import TaskDispatcher
from memopilot.runtime.tools import ToolRegistry
from memopilot.tasks.agent_task import AgentTask

NOW = datetime(2026, 7, 30, tzinfo=UTC)
async def test_agent_core_process_delegates_inbound_message_to_pipeline() -> None:
    class Pipeline:
        def __init__(self) -> None:
            self.message: InboundMessage | None = None
            self.key = ""

        async def run(
            self,
            message: InboundMessage,
            key: str,
            *,
            dispatch_outbound: bool = True,
        ) -> str:
            assert dispatch_outbound is True
            self.message = message
            self.key = key
            return "sent"

    message = InboundMessage("cli", "user", "chat", "hello", timestamp=NOW)
    pipeline = Pipeline()
    core = AgentCore(pipeline)  # type: ignore[arg-type]

    assert await core.process(message, message.session_key) == "sent"
    assert pipeline.message is message
    assert pipeline.key == message.session_key


class _Provider(ChatProvider):
    async def complete(
        self, *, messages: tuple[ChatMessage, ...], tools: tuple[ToolSchema, ...]
    ) -> ModelResponse:
        del messages, tools
        return ModelResponse(content="reply", tool_calls=(), finish_reason="stop")


class _Runtime(DefaultReasoner):
    def __init__(self, *, modules: tuple[object, ...] = ()) -> None:
        super().__init__(_Provider(), ToolRegistry(), modules=modules)  # type: ignore[arg-type]
        self.calls = 0
        self.standalone_calls = 0

    async def run(self, *args: object, **kwargs: object) -> ReActResult:
        self.standalone_calls += 1
        raise AssertionError("direct turn must use the passive pipeline")

    async def run_reasoning(self, *args: object, **kwargs: object) -> ReActResult:
        self.calls += 1
        return await super().run_reasoning(*args, **kwargs)  # type: ignore[arg-type]


class _Outbound:
    def __init__(self) -> None:
        self.sent: list[OutboundDispatch] = []
        self.ok = False

    async def dispatch(self, dispatch: OutboundDispatch) -> bool:
        self.sent.append(dispatch)
        return self.ok


class _AfterTurnProbe:
    phase = LifecyclePhase.AFTER_TURN
    slot = "test.after_turn_probe"
    requires = ("after_turn.build_ctx", "turn:ctx")
    produces: tuple[str, ...] = ()

    def __init__(self, *, mutate: bool = False) -> None:
        self.calls = 0
        self.exit_reasons: list[str] = []
        self.mutate = mutate

    async def run(self, context: PhaseContext) -> dict[str, object]:
        self.calls += 1
        phase_input = context.slots["after_turn.input"]
        assert isinstance(phase_input, AfterTurnInput)
        self.exit_reasons.append(phase_input.result.turn_result.exit_reason)
        ctx = context.slots["turn:ctx"]
        assert isinstance(ctx, AfterTurnCtx)
        if self.mutate:
            ctx.reply = "plugin reply"
            ctx.media = (*ctx.media, "plugin.png")
            ctx.outbound_metadata["plugin"] = "kept"
        return {}


class _UnusedHandler:
    async def execute_task(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        raise AssertionError()


@pytest.mark.asyncio
async def test_before_turn_abort_skips_reasoning_commit_and_memory_tasks(tmp_path) -> None:
    class Abort:
        phase = LifecyclePhase.BEFORE_TURN
        slot = "test.abort"
        requires = ("before_turn.build_ctx", "session:ctx")
        produces: tuple[str, ...] = ()

        async def run(self, context: PhaseContext) -> dict[str, object]:
            ctx = context.slots["session:ctx"]
            ctx.abort = True
            ctx.abort_reply = "stopped"
            return {}

    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = ConversationRepository(database)
    runtime, outbound = _Runtime(modules=(Abort(),)), _Outbound()
    outbound.ok = True
    dispatcher = TaskDispatcher(
        passive=AgentCore(
            PassiveTurnPipeline(
                runtime,
                repository=repository,
                outbound=outbound,
                event_bus=EventBus(),
                history_limit=12,
                outer_modules=(Abort(),),
            )
        ),
        memory=_UnusedHandler(),  # type: ignore[arg-type]
        proactive=_UnusedHandler(),  # type: ignore[arg-type]
        scheduler=_UnusedHandler(),  # type: ignore[arg-type]
    )
    task = AgentTask(
        "t",
        "passive.turn",
        0,
        "cli:chat",
        {
            "channel": "cli",
            "sender": "user",
            "chat_id": "chat",
            "content": "hi",
            "timestamp": NOW.isoformat(),
            "metadata": {"message_id": "abort"},
        },
        NOW,
    )

    assert await dispatcher.dispatch(task, now=NOW) == ()
    assert runtime.calls == 0
    assert outbound.sent[0].content == "stopped"
    assert repository.list_recent_messages("cli:chat", limit=10) == ()


@pytest.mark.asyncio
async def test_before_reasoning_abort_stops_dispatcher_without_commit_or_memory(tmp_path) -> None:
    class Abort:
        phase = LifecyclePhase.BEFORE_REASONING
        slot = "test.before_reasoning_abort"
        requires = ("before_reasoning.build_ctx", "reasoning:ctx")
        produces: tuple[str, ...] = ()

        async def run(self, context: PhaseContext) -> dict[str, object]:
            context.slots["reasoning:ctx"].abort = True
            context.slots["reasoning:ctx"].abort_reply = "reasoning stopped"
            return {}

    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository, outbound = ConversationRepository(database), _Outbound()
    outbound.ok = True
    runtime = _Runtime()
    dispatcher = TaskDispatcher(
        passive=AgentCore(PassiveTurnPipeline(
            runtime, repository=repository, outbound=outbound, event_bus=EventBus(),
            history_limit=12, outer_modules=(Abort(),),
        )),
        memory=_UnusedHandler(), proactive=_UnusedHandler(), scheduler=_UnusedHandler(),  # type: ignore[arg-type]
    )
    task = AgentTask("abort", "passive.turn", 0, "cli:chat", {
        "channel": "cli", "sender": "user", "chat_id": "chat", "content": "hi",
        "timestamp": NOW.isoformat(), "metadata": {"message_id": "before-reasoning"},
    }, NOW)

    assert await dispatcher.dispatch(task, now=NOW) == ()
    assert runtime.calls == 0
    assert [item.content for item in outbound.sent] == ["reasoning stopped"]
    assert repository.list_recent_messages("cli:chat", limit=10) == ()


@pytest.mark.asyncio
async def test_pipeline_does_not_replay_committed_turns(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = ConversationRepository(database)
    message = InboundMessage(
        "cli", "user", "chat", "hi", timestamp=NOW, metadata={"message_id": "m1"}
    )
    repository.record_inbound_activity(message)
    repository.commit_turn(message, assistant_content="old reply")
    outbound = _Outbound()
    pipeline = PassiveTurnPipeline(
        _Runtime(),
        repository=repository,
        outbound=outbound,
        event_bus=EventBus(),
        history_limit=12,
    )
    outbound.ok = True
    repository.find_committed_turn = lambda _message: (_ for _ in ()).throw(AssertionError())  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="助手消息内容不一致"):
        await pipeline.run(message, message.session_key)

    assert pipeline._runtime.calls == 1
    assert outbound.sent == []


@pytest.mark.asyncio
async def test_after_reasoning_commits_before_after_turn(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = ConversationRepository(database)
    message = InboundMessage(
        "cli", "user", "chat", "hi", timestamp=NOW, metadata={"message_id": "m1"}
    )
    repository.record_inbound_activity(message)
    outbound = _Outbound()
    outbound.ok = True
    pipeline = PassiveTurnPipeline(
        _Runtime(), repository=repository, outbound=outbound, event_bus=EventBus(), history_limit=12
    )
    class Probe(_AfterTurnProbe):
        async def run(self, context: PhaseContext) -> dict[str, object]:
            assert repository.find_committed_turn(message) is not None
            return await super().run(context)

    pipeline = PassiveTurnPipeline(
        _Runtime(),
        repository=repository,
        outbound=outbound,
        event_bus=EventBus(),
        history_limit=12,
        after_turn_modules=(Probe(),),
    )
    await pipeline.run(message, message.session_key)

    assert repository.find_committed_turn(message) is not None


@pytest.mark.asyncio
async def test_direct_replay_uses_stable_source_ref_and_runs_after_turn(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    runtime, outbound = _Runtime(), _Outbound()
    probe = _AfterTurnProbe()
    repository = ConversationRepository(database)
    pipeline = PassiveTurnPipeline(
        runtime,
        repository=repository,
        outbound=outbound,
        event_bus=EventBus(),
        history_limit=12,
        after_turn_modules=(probe,),
    )
    first = TurnInput("scheduler:e", "go", received_at=NOW, memory_source_ref="task:t")
    second = TurnInput(
        "scheduler:e", "go", received_at=NOW.replace(year=2027), memory_source_ref="task:t"
    )

    await pipeline.execute_direct(first)
    await pipeline.execute_direct(second)

    assert runtime.calls == 1
    assert runtime.standalone_calls == 0
    assert probe.calls == 2
    assert probe.exit_reasons == ["completed", "replay"]
    assert outbound.sent == []
    history = repository.list_recent_messages("scheduler:e", limit=10)
    assert [(item.role, item.content) for item in history] == [("assistant", "reply")]


@pytest.mark.asyncio
async def test_after_turn_plugin_changes_dispatch_and_sees_real_react(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    runtime, outbound, bus = _Runtime(), _Outbound(), EventBus()
    outbound.ok = True
    committed_events: list[TurnCommitted] = []

    async def capture(event: TurnCommitted) -> None:
        committed_events.append(event)

    bus.on(TurnCommitted, capture, observer=True)
    probe = _AfterTurnProbe(mutate=True)
    pipeline = PassiveTurnPipeline(
        runtime,
        repository=ConversationRepository(database),
        outbound=outbound,
        event_bus=bus,
        history_limit=12,
        after_turn_modules=(probe,),
    )
    await pipeline.run(
        InboundMessage(
            "cli", "user", "chat", "hi", timestamp=NOW, metadata={"message_id": "m-plugin"}
        ),
        "cli:chat",
    )

    assert probe.calls == 1
    assert probe.exit_reasons == ["completed"]
    assert len(committed_events) == 1
    assert len(outbound.sent) == 1
    assert outbound.sent[0].content == "plugin reply"
    assert outbound.sent[0].media == ["plugin.png"]
    assert outbound.sent[0].metadata["plugin"] == "kept"


@pytest.mark.asyncio
async def test_after_turn_orders_committed_event_then_plugin_then_dispatch(tmp_path) -> None:
    order: list[str] = []

    class Outbound(_Outbound):
        async def dispatch(self, dispatch: OutboundDispatch) -> bool:
            order.append("dispatch")
            return await super().dispatch(dispatch)

    class Plugin(_AfterTurnProbe):
        async def run(self, context: PhaseContext) -> dict[str, object]:
            order.append("plugin")
            return await super().run(context)

    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    bus, outbound = EventBus(), Outbound()
    outbound.ok = True

    async def committed(_: TurnCommitted) -> None:
        order.append("committed")

    bus.on(TurnCommitted, committed, observer=True)
    pipeline = PassiveTurnPipeline(
        _Runtime(), repository=ConversationRepository(database), outbound=outbound,
        event_bus=bus, history_limit=12, after_turn_modules=(Plugin(),),
    )
    await pipeline.run(
        InboundMessage(
            "cli", "user", "chat", "hi", timestamp=NOW, metadata={"message_id": "order"}
        ),
        "cli:chat",
    )

    assert order == ["committed", "plugin", "dispatch"]


@pytest.mark.asyncio
async def test_passive_provider_receives_replayed_proactive_context(tmp_path) -> None:
    class Provider(_Provider):
        def __init__(self) -> None:
            self.messages: tuple[ChatMessage, ...] = ()

        async def complete(self, *, messages, tools):  # type: ignore[no-untyped-def]
            del tools
            self.messages = messages
            return ModelResponse(content="reply", tool_calls=(), finish_reason="stop")

    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = ConversationRepository(database)
    repository.commit_direct_assistant(
        session_key="cli:chat",
        channel="cli",
        chat_id="chat",
        content="这篇文章讲 SQLite",
        media=(),
        timestamp=NOW,
        source_ref="proactive:decision-1",
        metadata={
            "proactive": True,
            "state_summary_tag": "none",
        },
    )
    provider, outbound = Provider(), _Outbound()
    outbound.ok = True
    pipeline = PassiveTurnPipeline(
        DefaultReasoner(provider, ToolRegistry()),
        repository=repository,
        outbound=outbound,
        event_bus=EventBus(),
        history_limit=12,
    )

    message = InboundMessage("cli", "user", "chat", "刚才那篇讲什么", timestamp=NOW)
    await pipeline.run(message, "cli:chat")

    contents = [str(message.content) for message in provider.messages]
    assert "[主动推送] 这篇文章讲 SQLite" in contents
    assert not any("recent_proactive_message_meta" in content for content in contents)
