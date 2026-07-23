from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.runtime.engine import TurnInput
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.scheduling.contracts import CreateSchedule
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import InboundCommand, OperationalRepository
from memopilot.tasks.outbox import OutboxDispatcher
from memopilot.tasks.redis_queue import RedisTaskQueue
from memopilot.worker.service import WorkerService

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[Redis]:
    url = os.getenv("MEMOPILOT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
    client = Redis.from_url(url, decode_responses=True)
    await client.ping()
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


def _repository(tmp_path: Path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    with connect_database(database) as connection:
        now = NOW.isoformat()
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-1', 'feishu', 'chat-1', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO session_activity(session_key, activity_version, updated_at) "
            "VALUES ('feishu:chat-1', 0, ?)",
            (now,),
        )
    return OperationalRepository(database)


class _UnusedRuntime:
    async def run(self, turn: TurnInput, **kwargs: Any):
        raise AssertionError("阻塞系统任务不应进入 Agent Runtime")


class _BlockingSystemJobs:
    def __init__(self, repository: OperationalRepository) -> None:
        self.repository = repository
        self.started = asyncio.Event()
        self.cancelled = False

    async def execute(self, *, kind: str, payload: dict[str, object], **kwargs: Any):
        if kind == "schedule.run":
            self.repository.transition_scheduled_execution(
                str(payload["execution_id"]),
                job_id=kwargs["claim"].job_id,
                lease=kwargs["lease"],
                outcome="running",
                now=kwargs["now"],
            )
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def _worker(
    repository: OperationalRepository,
    redis_client: Redis,
    system_jobs: _BlockingSystemJobs,
) -> tuple[WorkerService, RedisTaskQueue]:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    executor = RuntimeJobExecutor(
        repository,
        _UnusedRuntime(),  # type: ignore[arg-type]
        system_jobs=system_jobs,  # type: ignore[arg-type]
        clock=lambda: NOW,
    )
    return (
        WorkerService(
            repository,
            queue,
            SessionLeaseManager(redis_client, repository, ttl=timedelta(seconds=2)),
            executor,
            owner_id="worker-1",
            clock=lambda: NOW,
            heartbeat_interval=0.05,
            interrupt_poll_interval=0.01,
        ),
        queue,
    )


def _accept_new_user_activity(repository: OperationalRepository) -> None:
    repository.accept_inbound(
        InboundCommand(
            event_id="event-p0",
            message_id="message-p0",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "优先处理我"},
            received_at=NOW + timedelta(seconds=1),
        )
    )


@pytest.mark.asyncio
async def test_p1_schedule_is_atomically_requeued_and_old_redis_message_is_acked(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository = _repository(tmp_path)
    schedules = ScheduleRepository(repository.database)
    task = schedules.create(
        CreateSchedule(
            session_key="feishu:chat-1",
            schedule_kind="after",
            schedule_expression="1s",
            execution_mode="instant",
            payload={"message": "提醒"},
            next_run_at=NOW,
            timezone="UTC",
            created_at=NOW,
        )
    )
    execution = schedules.enqueue_due(now=NOW).queued[0]
    blocking = _BlockingSystemJobs(repository)
    worker, queue = await _worker(repository, redis_client, blocking)
    outbox = OutboxDispatcher(repository, queue, owner_id="app")
    assert await outbox.dispatch_one(now=NOW)

    running = asyncio.create_task(worker.run_once())
    await asyncio.wait_for(blocking.started.wait(), timeout=2)
    _accept_new_user_activity(repository)
    assert await asyncio.wait_for(running, timeout=2) is True

    job = repository.get_job(execution.job_id or "")
    assert blocking.cancelled is True
    assert job is not None and job.state == "queued"
    assert job.activity_version == 1
    with connect_database(repository.database) as connection:
        assert connection.execute(
            "SELECT state FROM scheduled_executions WHERE task_id = ?",
            (task.task_id,),
        ).fetchone()[0] == "queued"
        assert connection.execute(
            "SELECT state FROM runs WHERE job_id = ?", (job.job_id,)
        ).fetchone()[0] == "recovering"
        assert connection.execute(
            "SELECT COUNT(*) FROM outbox_events "
            "WHERE aggregate_id = ? AND state = 'pending'",
            (job.job_id,),
        ).fetchone()[0] == 1
    assert await redis_client.xlen(queue.stream_key(1)) == 0
    assert await outbox.dispatch_one(now=NOW + timedelta(seconds=2))
    entries = await redis_client.xrange(queue.stream_key(1))
    assert [fields["job_id"] for _, fields in entries] == [job.job_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "priority"), [("proactive.tick", 2), ("drift.run", 3)])
async def test_p2_p3_are_cancelled_on_new_activity_and_not_requeued(
    tmp_path: Path,
    redis_client: Redis,
    kind: str,
    priority: int,
) -> None:
    repository = _repository(tmp_path)
    result = repository.enqueue_system_job(
        kind=kind,
        priority=priority,
        session_key="feishu:chat-1",
        idempotency_key=f"{kind}:1",
        activity_version=0,
        payload={"chat_id": "chat-1"},
        now=NOW,
    )
    blocking = _BlockingSystemJobs(repository)
    worker, queue = await _worker(repository, redis_client, blocking)
    outbox = OutboxDispatcher(repository, queue, owner_id="app")
    assert await outbox.dispatch_one(now=NOW)

    running = asyncio.create_task(worker.run_once())
    await asyncio.wait_for(blocking.started.wait(), timeout=2)
    _accept_new_user_activity(repository)
    assert await asyncio.wait_for(running, timeout=2) is True

    job = repository.get_job(result.job_id)
    assert blocking.cancelled is True
    assert job is not None and job.state == "cancelled"
    assert await redis_client.xlen(queue.stream_key(priority)) == 0
    with connect_database(repository.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM outbox_events "
            "WHERE aggregate_id = ? AND state = 'pending'",
            (job.job_id,),
        ).fetchone()[0] == 0
