from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.runtime.agent_loop import AgentLoop
from memopilot.runtime.outbound import DeliveryError

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
        self.published: list[object] = []

    async def read_next(self, *, consumer_id: str) -> object | None:
        del consumer_id
        message, self.message = self.message, None
        return message

    async def acknowledge(self, message: object) -> None:
        del message
        self.acked = True

    async def publish_task_once(self, task: object) -> str:
        self.published.append(task)
        return "2-0"


class _Leases:
    ttl_ms = 30_000

    async def acquire(self, session_key: str, *, owner_id: str, now: datetime) -> object:
        del now
        return SimpleNamespace(
            session_key=session_key,
            owner_id=owner_id,
            epoch=1,
            redis_key="lease",
            redis_value="value",
        )

    async def renew(self, lease: object, *, now: datetime) -> bool:
        del lease, now
        return True

    async def release(self, lease: object) -> bool:
        del lease
        return True


class _Coordinator:
    def __init__(self, *, reason: str | None = None) -> None:
        self.reason = reason

    async def clear_background_stop(self, session_key: str) -> None:
        del session_key
        self.reason = None

    async def stop_reason(self, session_key: str) -> str | None:
        del session_key
        return self.reason


class _Dispatcher:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.cancelled = False
        self.calls = 0

    async def dispatch(self, *args: Any, **kwargs: Any) -> tuple[()]:
        del args, kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ()


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
        _Leases(),  # type: ignore[arg-type]
        runner,
        owner_id="agent-1",
        session_coordinator=_Coordinator(),  # type: ignore[arg-type]
    )

    assert await loop.run_once() is True
    assert runner.calls == 1
    assert queue.acked is True


@pytest.mark.asyncio
async def test_send_failure_is_not_acked() -> None:
    queue = _Queue()
    loop = AgentLoop(
        queue,  # type: ignore[arg-type]
        _Leases(),  # type: ignore[arg-type]
        _Dispatcher(DeliveryError("network down")),
        owner_id="agent-1",
        session_coordinator=_Coordinator(),  # type: ignore[arg-type]
    )

    with pytest.raises(DeliveryError):
        await loop.run_once()
    assert queue.acked is False


@pytest.mark.asyncio
async def test_preempted_schedule_stays_pending_without_ack_or_republish() -> None:
    queue = _Queue(kind="schedule.run")
    runner = _BlockingDispatcher()
    loop = AgentLoop(
        queue,  # type: ignore[arg-type]
        _Leases(),  # type: ignore[arg-type]
        runner,
        owner_id="agent-1",
        session_coordinator=_Coordinator(reason="user_message"),  # type: ignore[arg-type]
        interrupt_poll_interval=0.001,
    )

    assert await loop.run_once() is True
    assert runner.cancelled is True
    assert queue.acked is False
    assert queue.published == []


@pytest.mark.asyncio
async def test_user_stop_cancels_and_acks_running_passive_turn() -> None:
    queue = _Queue()
    runner = _BlockingDispatcher()
    coordinator = _Coordinator()
    loop = AgentLoop(
        queue,  # type: ignore[arg-type]
        _Leases(),  # type: ignore[arg-type]
        runner,
        owner_id="agent-1",
        session_coordinator=coordinator,  # type: ignore[arg-type]
        interrupt_poll_interval=0.001,
    )

    running = asyncio.create_task(loop.run_once())
    await runner.started.wait()
    coordinator.reason = "user_stop"

    assert await running is True
    assert runner.cancelled is True
    assert queue.acked is True
    assert coordinator.reason is None
