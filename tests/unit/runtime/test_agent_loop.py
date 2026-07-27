from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.bus.queue import MessageBus
from memopilot.runtime.agent_loop import AgentLoop, _explicitly_memorized_ids
from memopilot.runtime.engine import TurnResult
from memopilot.runtime.react import ReActResult
from memopilot.tasks.background import BackgroundTask


class _Runtime:
    async def run(self, turn, **kwargs):
        return TurnResult(
            "你好呀。",
            (),
            ReActResult(
                reply="你好呀。",
                messages=(),
                iterations=1,
                tool_chain=(),
                exit_reason="completed",
                thinking="先接住",
            ),
            (),
            (),
        )


class _Store:
    def __init__(self, tasks: tuple[BackgroundTask, ...] = ()) -> None:
        self.committed: list[tuple[InboundMessage, str]] = []
        self.tasks = tasks

    def list_recent_messages(self, session_key: str, *, limit: int):
        return ()

    def record_inbound_activity(self, message):
        return 1

    async def commit_turn(
        self,
        message,
        *,
        assistant_content,
        cited_memory_ids,
        explicitly_memorized_ids,
    ):
        self.committed.append((message, assistant_content))
        return self.tasks


class _Publisher:
    def __init__(self) -> None:
        self.tasks: list[BackgroundTask] = []

    async def publish_task_once(self, task: BackgroundTask) -> str:
        self.tasks.append(task)
        return "1-0"


class _Coordinator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def request_background_stop(self, session_key: str, *, reason: str) -> None:
        self.events.append("stop-background")

    async def begin_user_turn(self, session_key: str, *, turn_id: str) -> None:
        self.events.append("begin-user")

    async def end_user_turn(self, session_key: str, *, turn_id: str) -> None:
        self.events.append("end-user")

    async def clear_background_stop(self, session_key: str) -> None:
        self.events.append("clear-stop")


class _OrderedBus(MessageBus):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def publish_outbound(self, message) -> None:
        self.events.append("reply")
        await super().publish_outbound(message)


class _OrderedPublisher(_Publisher):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def publish_task_once(self, task: BackgroundTask) -> str:
        self.events.append("background-task")
        return await super().publish_task_once(task)


def test_agent_loop_extracts_successful_memorize_results_without_step_table() -> None:
    trace = (
        SimpleNamespace(
            tool_name="memorize",
            state="succeeded",
            observation={"result": {"item_id": "mem-new"}},
        ),
        SimpleNamespace(
            tool_name="memorize",
            state="failed",
            observation={"result": {"item_id": "mem-failed"}},
        ),
        SimpleNamespace(
            tool_name="shell",
            state="succeeded",
            observation={"result": {"item_id": "not-memory"}},
        ),
    )

    assert _explicitly_memorized_ids(trace) == ("mem-new",)


@pytest.mark.asyncio
async def test_agent_loop_keeps_user_priority_until_reply_and_task_publication_finish() -> None:
    events: list[str] = []
    bus = _OrderedBus(events)
    task = BackgroundTask(
        "memory-1",
        "memory.consolidate",
        3,
        "feishu:chat-1",
        {},
        datetime(2026, 7, 27, tzinfo=UTC),
    )
    loop = AgentLoop(
        bus=bus,
        runtime=_Runtime(),
        store=_Store((task,)),
        session_coordinator=_Coordinator(events),  # type: ignore[arg-type]
        background_publisher=_OrderedPublisher(events),
    )

    await loop.process(
        InboundMessage(
            "feishu",
            "user",
            "chat-1",
            "你好",
            metadata={"message_id": "msg-order"},
        )
    )

    assert events == [
        "stop-background",
        "begin-user",
        "reply",
        "background-task",
        "end-user",
        "clear-stop",
    ]


@pytest.mark.asyncio
async def test_agent_loop_copies_prototype_bus_to_runtime_to_bus_flow() -> None:
    bus = MessageBus()
    store = _Store()
    loop = AgentLoop(bus=bus, runtime=_Runtime(), store=store)
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "你好",
        timestamp=datetime(2026, 7, 27, tzinfo=UTC),
        metadata={"message_id": "msg-1"},
    )

    await bus.publish_inbound(message)
    await loop.process(await bus.consume_inbound())
    outbound = await asyncio.wait_for(bus._outbound.get(), timeout=0.1)

    assert outbound.channel == "feishu"
    assert outbound.content == "你好呀。"
    assert outbound.thinking == "先接住"
    assert store.committed == [(message, "你好呀。")]


@pytest.mark.asyncio
async def test_agent_loop_publishes_memory_tasks_returned_by_turn_store() -> None:
    bus = MessageBus()
    task = BackgroundTask(
        "memory-1",
        "memory.consolidate",
        3,
        "feishu:chat-1",
        {"turn_id": "turn-1"},
        datetime(2026, 7, 27, tzinfo=UTC),
    )
    store = _Store((task,))
    publisher = _Publisher()
    loop = AgentLoop(
        bus=bus,
        runtime=_Runtime(),
        store=store,
        background_publisher=publisher,
    )

    await loop.process(
        InboundMessage(
            "feishu",
            "user",
            "chat-1",
            "你好",
            timestamp=datetime(2026, 7, 27, tzinfo=UTC),
            metadata={"message_id": "msg-memory"},
        )
    )

    assert publisher.tasks == [task]


class _BlockingRuntime:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.released = asyncio.Event()

    async def run(self, turn, **kwargs):
        self.started.set()
        await self.released.wait()
        return await _Runtime().run(turn, **kwargs)


@pytest.mark.asyncio
async def test_agent_loop_stop_cancels_active_turn_without_persistent_interrupt_job() -> None:
    bus = MessageBus()
    store = _Store()
    runtime = _BlockingRuntime()
    loop = AgentLoop(bus=bus, runtime=runtime, store=store)
    message = InboundMessage("feishu", "user", "chat-1", "/long", metadata={"message_id": "msg-2"})

    running = asyncio.create_task(loop.process(message))
    await runtime.started.wait()
    acknowledgement = await loop.request_interrupt(
        InboundMessage("feishu", "user", "chat-1", "/stop", metadata={"message_id": "stop-1"})
    )
    await running

    assert "中断" in acknowledgement.message
    assert store.committed == []
