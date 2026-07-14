from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.tasks.operational import LostLeaseError
from memopilot.worker.service import WorkerService

NOW = datetime(2026, 7, 14, tzinfo=UTC)


class _Queue:
    async def read_next(self, *, consumer_id: str) -> Any:
        return SimpleNamespace(job_id="job-1", session_key="feishu:chat-1")

    async def acknowledge(self, message: Any) -> None:
        raise AssertionError("失去 Lease 时不能 ACK")


class _Leases:
    ttl_ms = 30_000

    async def acquire(self, session_key: str, *, owner_id: str, now: datetime) -> Any:
        return SimpleNamespace(
            session_key=session_key,
            owner_id=owner_id,
            epoch=1,
            redis_key="lease",
            redis_value="value",
        )

    async def renew(self, lease: Any, *, now: datetime | None = None) -> bool:
        return False

    async def release(self, lease: Any) -> bool:
        return False


class _Repository:
    def claim_job(self, job_id: str, *, lease: Any, now: datetime) -> Any:
        return SimpleNamespace(
            run_id="run-1",
            job_id=job_id,
            session_key=lease.session_key,
            owner_id=lease.owner_id,
            fencing_epoch=lease.epoch,
        )

    def get_job(self, job_id: str) -> Any:
        return SimpleNamespace(payload_json='{"text":"你好"}', state="running")

    def list_recent_messages(self, session_key: str, *, limit: int) -> tuple[Any, ...]:
        return ()


class _Executor:
    cancelled = False

    async def execute(self, **kwargs: Any) -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _RenewingLeases(_Leases):
    async def renew(self, lease: Any, *, now: datetime | None = None) -> bool:
        return True


class _HeartbeatRepository(_Repository):
    def __init__(self) -> None:
        self.heartbeats = 0

    def heartbeat_run(self, run_id: str, *, lease: Any, now: datetime) -> None:
        assert run_id == "run-1"
        self.heartbeats += 1


class _CompletingExecutor:
    async def execute(self, **kwargs: Any) -> None:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_lease_renewal_failure_cancels_execution_without_ack() -> None:
    executor = _Executor()
    worker = WorkerService(
        _Repository(),  # type: ignore[arg-type]
        _Queue(),  # type: ignore[arg-type]
        _Leases(),  # type: ignore[arg-type]
        executor,  # type: ignore[arg-type]
        owner_id="worker-1",
        clock=lambda: NOW,
        heartbeat_interval=0.001,
    )

    with pytest.raises(LostLeaseError, match="续租失败"):
        await worker.run_once()

    assert executor.cancelled is True


@pytest.mark.asyncio
async def test_successful_lease_renewal_also_refreshes_run_heartbeat() -> None:
    repository = _HeartbeatRepository()
    worker = WorkerService(
        repository,  # type: ignore[arg-type]
        _Queue(),  # type: ignore[arg-type]
        _RenewingLeases(),  # type: ignore[arg-type]
        _CompletingExecutor(),  # type: ignore[arg-type]
        owner_id="worker-1",
        clock=lambda: NOW,
        heartbeat_interval=0.001,
    )

    assert await worker.run_once() is True
    assert repository.heartbeats > 0


def test_turn_input_loads_short_term_history_before_current_message() -> None:
    class RepositoryWithHistory(_Repository):
        def list_recent_messages(self, session_key: str, *, limit: int) -> tuple[Any, ...]:
            assert session_key == "feishu:chat-1"
            assert limit == 12
            return (
                SimpleNamespace(role="user", content="上一问"),
                SimpleNamespace(role="assistant", content="上一答"),
            )

    worker = WorkerService(
        RepositoryWithHistory(),  # type: ignore[arg-type]
        _Queue(),  # type: ignore[arg-type]
        _Leases(),  # type: ignore[arg-type]
        _CompletingExecutor(),  # type: ignore[arg-type]
        owner_id="worker-1",
        clock=lambda: NOW,
        short_term_message_limit=12,
    )
    claim = SimpleNamespace(session_key="feishu:chat-1")
    message = SimpleNamespace(job_id="job-1")

    turn = worker._turn_input(message, claim)

    assert [(item.role, item.content) for item in turn.history] == [
        ("user", "上一问"),
        ("assistant", "上一答"),
    ]
    assert turn.content == "你好"
