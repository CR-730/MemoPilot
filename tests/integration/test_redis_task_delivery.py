from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.operational import InboundCommand, OperationalRepository
from memopilot.tasks.outbox import OutboxDispatcher, QueueReconciler
from memopilot.tasks.redis_queue import PublishedJob, RedisTaskQueue

NOW = datetime(2026, 7, 13, 9, 0, tzinfo=UTC)


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


def make_repository(tmp_path: Path) -> tuple[OperationalRepository, str]:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    result = repository.accept_inbound(
        InboundCommand(
            event_id="event-1",
            message_id="message-1",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "你好"},
            received_at=NOW,
        )
    )
    return repository, result.job_id


@pytest.mark.asyncio
async def test_outbox_dispatches_to_stream_and_records_mirror(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    dispatcher = OutboxDispatcher(repository, queue, owner_id="app-1")

    dispatched = await dispatcher.dispatch_one(now=NOW)

    assert dispatched is True
    assert repository.get_outbox_for_job(job_id).state == "published"
    assert await redis_client.xlen(queue.stream_key(0)) == 1
    assert await redis_client.sismember(queue.queued_job_ids_key, job_id)


@pytest.mark.asyncio
async def test_committed_outbox_survives_crash_before_xadd(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()

    def crash_before_publish(name: str) -> None:
        if name == "after_commit_before_publish":
            raise RuntimeError("模拟 commit 后、XADD 前崩溃")

    crashing = OutboxDispatcher(
        repository,
        queue,
        owner_id="app-dead",
        claim_ttl=timedelta(seconds=1),
        failpoint=crash_before_publish,
    )
    with pytest.raises(RuntimeError, match="XADD 前"):
        await crashing.dispatch_one(now=NOW)

    assert repository.get_outbox_for_job(job_id).state == "publishing"
    assert await redis_client.xlen(queue.stream_key(0)) == 0
    recovered = OutboxDispatcher(repository, queue, owner_id="app-new")
    assert await recovered.dispatch_one(now=NOW + timedelta(seconds=2)) is True
    assert await redis_client.xlen(queue.stream_key(0)) == 1


@pytest.mark.asyncio
async def test_xadd_before_sqlite_mark_can_be_retried_as_safe_duplicate(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()

    def crash_after_xadd(name: str) -> None:
        if name == "after_publish_before_mark":
            raise RuntimeError("模拟 XADD 后崩溃")

    crashing = OutboxDispatcher(
        repository,
        queue,
        owner_id="app-dead",
        claim_ttl=timedelta(seconds=1),
        failpoint=crash_after_xadd,
    )
    with pytest.raises(RuntimeError, match="XADD 后"):
        await crashing.dispatch_one(now=NOW)

    assert repository.get_outbox_for_job(job_id).state == "publishing"
    recovered = OutboxDispatcher(repository, queue, owner_id="app-new")
    assert await recovered.dispatch_one(now=NOW + timedelta(seconds=2)) is True

    entries = await redis_client.xrange(queue.stream_key(0))
    assert [fields["job_id"] for _, fields in entries] == [job_id, job_id]
    assert repository.get_outbox_for_job(job_id).state == "published"


@pytest.mark.asyncio
async def test_reconciler_restores_queued_job_after_redis_is_cleared(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    dispatcher = OutboxDispatcher(repository, queue, owner_id="app-1")
    assert await dispatcher.dispatch_one(now=NOW) is True

    await redis_client.flushdb()
    await queue.ensure_consumer_groups()
    reconciler = QueueReconciler(repository, queue)
    recovered = await reconciler.reconcile(now=NOW + timedelta(minutes=1))

    assert recovered == (job_id,)
    outbox = repository.get_outbox_for_job(job_id)
    assert outbox.state == "pending"
    assert outbox.recovery_count == 1
    assert await dispatcher.dispatch_one(now=NOW + timedelta(minutes=1)) is True
    assert await redis_client.sismember(queue.queued_job_ids_key, job_id)


@pytest.mark.asyncio
async def test_reconciler_does_not_count_never_published_outbox_as_recovery(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()

    recovered = await QueueReconciler(repository, queue).reconcile(now=NOW)

    assert recovered == ()
    outbox = repository.get_outbox_for_job(job_id)
    assert outbox.state == "pending"
    assert outbox.recovery_count == 0


@pytest.mark.asyncio
async def test_consumer_reads_highest_available_priority_first(redis_client: Redis) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish(PublishedJob("job-p3", "drift.run", 3, "feishu:chat", "{}"))
    await queue.publish(PublishedJob("job-p2", "proactive.tick", 2, "feishu:chat", "{}"))
    await queue.publish(PublishedJob("job-p1", "schedule.run", 1, "feishu:chat", "{}"))
    await queue.publish(PublishedJob("job-p0", "agent.turn", 0, "feishu:chat", "{}"))

    first = await queue.read_next(consumer_id="worker-1")
    second = await queue.read_next(consumer_id="worker-1")
    third = await queue.read_next(consumer_id="worker-1")
    fourth = await queue.read_next(consumer_id="worker-1")

    assert [first.job_id, second.job_id, third.job_id, fourth.job_id] == [
        "job-p0",
        "job-p1",
        "job-p2",
        "job-p3",
    ]
    for message in (first, second, third, fourth):
        await queue.acknowledge(message)
    assert await redis_client.scard(queue.queued_job_ids_key) == 0


@pytest.mark.asyncio
async def test_unfinished_stream_entries_are_never_trimmed_by_publish(redis_client: Redis) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()

    for index in range(10_100):
        await queue.publish(PublishedJob(f"job-{index}", "agent.turn", 0, "feishu:chat", "{}"))

    first = await redis_client.xrange(queue.stream_key(0), count=1)
    assert first[0][1]["job_id"] == "job-0"
