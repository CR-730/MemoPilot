from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.runtime.background_task_loop import BackgroundTaskLoop
from memopilot.tasks.operational import LostLeaseError, StaleActivityError

NOW = datetime(2026, 7, 27, tzinfo=UTC)


class _Queue:
    def __init__(self) -> None:
        self.acked = False
        self.message = SimpleNamespace(
            task_id="task-1",
            kind="memory.optimize",
            priority=3,
            session_key="system:memory",
            payload_json="{}",
        )

    async def read_next(self, *, consumer_id: str) -> Any:
        del consumer_id
        message, self.message = self.message, None
        return message

    async def acknowledge(self, message: Any) -> None:
        del message
        self.acked = True


class _EmptyQueue:
    def __init__(self) -> None:
        self.pending_scans = 0

    async def read_next(self, *, consumer_id: str) -> None:
        del consumer_id
        return None

    async def pending_entries(self, **kwargs: Any) -> tuple[()]:
        del kwargs
        self.pending_scans += 1
        return ()


class _BrokenQueue:
    async def ensure_consumer_groups(self) -> None:
        return None

    async def read_next(self, *, consumer_id: str) -> None:
        del consumer_id
        raise RuntimeError("redis unavailable")


class _Leases:
    ttl_ms = 30_000

    async def acquire(self, session_key: str, *, owner_id: str, now: datetime) -> Any:
        del now
        return SimpleNamespace(
            session_key=session_key,
            owner_id=owner_id,
            epoch=1,
            redis_key="lease",
            redis_value="value",
        )

    async def renew(self, lease: Any, *, now: datetime | None = None) -> bool:
        del lease, now
        return False

    async def release(self, lease: Any) -> bool:
        del lease
        return True


class _Executor:
    def __init__(self) -> None:
        self.cancelled = False

    async def execute(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _UnusedExecutor:
    async def execute(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("空队列不应执行任务")


class _ProactiveQueue(_Queue):
    def __init__(self) -> None:
        super().__init__()
        self.message.kind = "proactive.tick"
        self.requeued = False
        self.published = False

    async def acknowledge_requeued(self, message: Any) -> None:
        del message
        self.requeued = True

    async def publish(self, message: Any) -> str:
        del message
        self.published = True
        return "1-0"


class _Coordinator:
    def __init__(self, *, active: bool, stop_requested: bool = False) -> None:
        self.active = active
        self.stop_requested = stop_requested

    async def user_turn_active(self, session_key: str) -> bool:
        del session_key
        return self.active

    async def background_stop_requested(self, session_key: str) -> bool:
        del session_key
        return self.stop_requested


class _ImmediateLeases(_Leases):
    async def renew(self, lease: Any, *, now: datetime | None = None) -> bool:
        del lease, now
        return True


class _StaleProactiveExecutor:
    async def execute(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise StaleActivityError("用户活动版本已变化")


@pytest.mark.asyncio
async def test_stale_proactive_tick_is_acked_instead_of_left_pending() -> None:
    queue = _ProactiveQueue()
    loop = BackgroundTaskLoop(
        queue,  # type: ignore[arg-type]
        _ImmediateLeases(),  # type: ignore[arg-type]
        _StaleProactiveExecutor(),
        owner_id="runner-1",
    )

    assert await loop.run_once() is True
    assert queue.acked is True
    assert queue.requeued is False
    assert queue.published is False


@pytest.mark.asyncio
async def test_proactive_tick_seen_during_user_turn_is_dropped_until_next_tick() -> None:
    queue = _ProactiveQueue()
    executor = _Executor()
    loop = BackgroundTaskLoop(
        queue,  # type: ignore[arg-type]
        _ImmediateLeases(),  # type: ignore[arg-type]
        executor,
        owner_id="runner-1",
        session_coordinator=_Coordinator(active=True),  # type: ignore[arg-type]
    )

    assert await loop.run_once() is True
    assert queue.acked is True
    assert queue.requeued is False
    assert queue.published is False


@pytest.mark.asyncio
async def test_running_proactive_tick_preempted_by_user_waits_for_next_tick() -> None:
    queue = _ProactiveQueue()
    executor = _Executor()
    loop = BackgroundTaskLoop(
        queue,  # type: ignore[arg-type]
        _ImmediateLeases(),  # type: ignore[arg-type]
        executor,
        owner_id="runner-1",
        interrupt_poll_interval=0.001,
        session_coordinator=_Coordinator(  # type: ignore[arg-type]
            active=False,
            stop_requested=True,
        ),
    )

    assert await loop.run_once() is True
    assert executor.cancelled is True
    assert queue.acked is True
    assert queue.requeued is False
    assert queue.published is False


@pytest.mark.asyncio
async def test_lease_renewal_failure_cancels_task_without_ack() -> None:
    queue = _Queue()
    executor = _Executor()
    loop = BackgroundTaskLoop(
        queue,  # type: ignore[arg-type]
        _Leases(),  # type: ignore[arg-type]
        executor,
        owner_id="runner-1",
        clock=lambda: NOW,
        heartbeat_interval=0.001,
        interrupt_poll_interval=0.001,
    )

    with pytest.raises(LostLeaseError, match="续租失败"):
        await loop.run_once()

    assert executor.cancelled is True
    assert queue.acked is False


@pytest.mark.asyncio
async def test_empty_queue_throttles_pending_recovery_scans() -> None:
    queue = _EmptyQueue()
    monotonic_values = iter((0.0, 0.1, 0.2, 5.0))
    loop = BackgroundTaskLoop(
        queue,  # type: ignore[arg-type]
        _ImmediateLeases(),  # type: ignore[arg-type]
        _UnusedExecutor(),
        owner_id="runner-1",
        pending_scan_interval=5.0,
        monotonic=lambda: next(monotonic_values),
    )

    assert await loop.run_once() is False
    assert await loop.run_once() is False
    assert await loop.run_once() is False
    assert await loop.run_once() is False
    assert queue.pending_scans == 8


@pytest.mark.asyncio
async def test_consumer_error_uses_idle_backoff() -> None:
    delays: list[float] = []

    async def stop_after_sleep(delay: float) -> None:
        delays.append(delay)
        raise asyncio.CancelledError

    loop = BackgroundTaskLoop(
        _BrokenQueue(),  # type: ignore[arg-type]
        _ImmediateLeases(),  # type: ignore[arg-type]
        _UnusedExecutor(),
        owner_id="runner-1",
        sleep=stop_after_sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        await loop.run_forever(idle_interval=0.25)

    assert delays == [0.25]
