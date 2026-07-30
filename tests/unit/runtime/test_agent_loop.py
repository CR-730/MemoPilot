from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.runtime.agent_loop import AgentLoop
from memopilot.runtime.outbound import DeliveryError
from memopilot.tasks.agent_task import AgentTask

NOW = datetime(2026, 7, 28, tzinfo=UTC)


class _Queue:
    def __init__(self, *, kind: str = "passive.turn") -> None:
        self.message = SimpleNamespace(
            task_id="task-1",
            kind=kind,
            priority=0 if kind == "passive.turn" else 1,
            session_key="feishu:chat-1",
            payload_json=json.dumps({}),
        )
        self.acked = False
        self.acked_task_ids: list[str] = []
        self.published: list[object] = []
        self.events: list[str] = []

    async def read_next(self, *, consumer_id: str) -> object | None:
        del consumer_id
        message, self.message = self.message, None
        return message

    async def read_pending(self, *, consumer_id: str) -> object | None:
        del consumer_id
        return None

    async def read_priority(
        self, priority: int, *, consumer_id: str, entry_id: str = ">"
    ) -> object | None:
        del priority, consumer_id, entry_id
        return None

    async def acknowledge(self, message: object) -> None:
        self.acked = True
        self.acked_task_ids.append(str(message.task_id))
        self.events.append("ack")

    async def publish_task_once(self, task: object) -> str:
        self.published.append(task)
        self.events.append("publish")
        return "2-0"


class _Dispatcher:
    def __init__(
        self,
        error: BaseException | None = None,
        result: tuple[AgentTask, ...] = (),
    ) -> None:
        self.error = error
        self.result = result
        self.cancelled = False
        self.calls = 0

    async def dispatch(self, *args: Any, **kwargs: Any) -> tuple[AgentTask, ...]:
        del args, kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _BlockingDispatcher(_Dispatcher):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def dispatch(self, *args: Any, **kwargs: Any) -> tuple[()]:
        del args, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_passive_p0_is_executed_and_acked_by_only_agent_loop() -> None:
    queue = _Queue()
    runner = _Dispatcher()
    loop = AgentLoop(
        queue,  # type: ignore[arg-type]
        runner,
    )

    assert await loop.run_once() is True
    assert runner.calls == 1
    assert queue.acked is True


@pytest.mark.asyncio
async def test_send_failure_is_not_acked() -> None:
    queue = _Queue()
    loop = AgentLoop(
        queue,  # type: ignore[arg-type]
        _Dispatcher(DeliveryError("network down")),
    )

    with pytest.raises(DeliveryError):
        await loop.run_once()
    assert queue.acked is False


@pytest.mark.asyncio
async def test_pending_message_is_replayed_before_new_message() -> None:
    queue = _Queue(kind="schedule.run")
    pending = SimpleNamespace(
        task_id="pending-1",
        kind="memory.optimize",
        priority=3,
        session_key="system:memory",
        payload_json="{}",
    )

    async def read_pending(*, consumer_id: str) -> object | None:
        del consumer_id
        return pending

    queue.read_pending = read_pending  # type: ignore[method-assign]
    runner = _Dispatcher()
    loop = AgentLoop(
        queue,  # type: ignore[arg-type]
        runner,
    )

    assert await loop.run_once() is True
    assert runner.calls == 1
    assert queue.acked is True


@pytest.mark.asyncio
async def test_cancelled_task_is_left_pending() -> None:
    queue = _Queue()
    runner = _BlockingDispatcher()
    loop = AgentLoop(
        queue,  # type: ignore[arg-type]
        runner,
    )

    running = asyncio.create_task(loop.run_once())
    await runner.started.wait()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert runner.cancelled is True
    assert queue.acked is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, priority, expected_ack",
    [
        ("schedule.run", 1, False),
        ("proactive.tick", 2, True),
        ("memory.optimize", 3, False),
    ],
)
async def test_p0_preempts_background_task_with_priority_ack_policy(
    kind: str,
    priority: int,
    expected_ack: bool,
) -> None:
    queue = _Queue(kind=kind)
    queue.message.priority = priority
    p0 = SimpleNamespace(
        task_id="p0",
        kind="passive.turn",
        priority=0,
        session_key="feishu:chat-1",
        payload_json="{}",
    )

    async def read_priority(
        priority: int, *, consumer_id: str, entry_id: str = ">"
    ) -> object | None:
        del consumer_id, entry_id
        return p0 if priority == 0 else None

    queue.read_priority = read_priority  # type: ignore[method-assign]
    runner = _BlockingDispatcher()
    loop = AgentLoop(queue, runner, interrupt_poll_interval=0)  # type: ignore[arg-type]

    assert await loop.run_once() is True
    assert runner.cancelled is True
    assert queue.acked is expected_ack


@pytest.mark.asyncio
async def test_p0_is_not_preempted_by_another_p0() -> None:
    queue = _Queue()
    read_priority_calls = 0

    async def read_priority(
        priority: int, *, consumer_id: str, entry_id: str = ">"
    ) -> object | None:
        nonlocal read_priority_calls
        del priority, consumer_id, entry_id
        read_priority_calls += 1
        return None

    queue.read_priority = read_priority  # type: ignore[method-assign]
    loop = AgentLoop(queue, _Dispatcher())  # type: ignore[arg-type]

    assert await loop.run_once() is True
    assert read_priority_calls == 0
    assert queue.acked is True


@pytest.mark.asyncio
async def test_user_stop_cancels_and_acks_current_p0() -> None:
    queue = _Queue()
    dispatcher = _BlockingDispatcher()
    loop = AgentLoop(queue, dispatcher)  # type: ignore[arg-type]
    running = asyncio.create_task(loop.run_once())
    await dispatcher.started.wait()

    assert loop.cancel_current() is True
    assert await running is True
    assert dispatcher.cancelled is True
    assert queue.acked is True


@pytest.mark.asyncio
async def test_derived_tasks_are_published_before_source_ack() -> None:
    queue = _Queue()
    derived = AgentTask(
        "derived-1",
        "memory.post_response",
        3,
        "feishu:chat-1",
        {},
        NOW,
    )
    loop = AgentLoop(queue, _Dispatcher(result=(derived,)))  # type: ignore[arg-type]

    assert await loop.run_once() is True
    assert queue.published == [derived]
    assert queue.events == ["publish", "ack"]


@pytest.mark.asyncio
async def test_direct_p0_preemption_replays_and_executes_the_new_p0_next() -> None:
    queue = _Queue(kind="schedule.run")
    p0 = SimpleNamespace(
        task_id="p0",
        kind="passive.turn",
        priority=0,
        session_key="feishu:chat-1",
        payload_json="{}",
    )
    p0_pending = False

    async def read_pending(*, consumer_id: str) -> object | None:
        nonlocal p0_pending
        del consumer_id
        if not p0_pending:
            return None
        p0_pending = False
        return p0

    queue.read_pending = read_pending  # type: ignore[method-assign]
    blocking = _BlockingDispatcher()
    loop = AgentLoop(queue, blocking)  # type: ignore[arg-type]
    first = asyncio.create_task(loop.run_once())
    await blocking.started.wait()

    p0_pending = True
    assert loop.preempt_for_p0() is True
    assert await first is True
    assert queue.acked_task_ids == []

    loop._dispatcher = _Dispatcher()  # noqa: SLF001
    assert await loop.run_once() is True
    assert queue.acked_task_ids == ["p0"]
