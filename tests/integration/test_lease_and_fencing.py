from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import (
    InboundCommand,
    LostLeaseError,
    OperationalRepository,
)
from memopilot.tasks.recovery import PendingDisposition, PendingMessageReclaimer
from memopilot.tasks.redis_queue import PublishedJob, RedisTaskQueue

NOW = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)


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
async def test_lease_is_exclusive_and_epoch_monotonically_increases(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, _ = make_repository(tmp_path)
    leases = SessionLeaseManager(redis_client, repository, ttl=timedelta(seconds=30))

    first = await leases.acquire("feishu:chat-1", owner_id="worker-a", now=NOW)
    blocked = await leases.acquire("feishu:chat-1", owner_id="worker-b", now=NOW)

    assert first is not None
    assert blocked is None
    assert first.epoch == 1
    assert await leases.release(first) is True
    second = await leases.acquire("feishu:chat-1", owner_id="worker-b", now=NOW)
    assert second is not None
    assert second.epoch == 2


@pytest.mark.asyncio
async def test_stale_worker_cannot_renew_release_or_write_after_takeover(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    leases = SessionLeaseManager(redis_client, repository, ttl=timedelta(milliseconds=150))
    stale = await leases.acquire("feishu:chat-1", owner_id="worker-a", now=NOW)
    assert stale is not None
    run = repository.claim_job(job_id, lease=stale, now=NOW)
    assert run is not None

    await asyncio.sleep(0.2)
    current = await leases.acquire(
        "feishu:chat-1",
        owner_id="worker-b",
        now=NOW + timedelta(seconds=1),
    )
    assert current is not None
    assert current.epoch > stale.epoch

    assert await leases.renew(stale) is False
    assert await leases.release(stale) is False
    assert await leases.renew(current) is True
    with pytest.raises(LostLeaseError):
        repository.heartbeat_run(run.run_id, lease=stale, now=NOW + timedelta(seconds=1))


@pytest.mark.asyncio
async def test_duplicate_delivery_reuses_run_and_recovery_creates_new_attempt(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    leases = SessionLeaseManager(redis_client, repository, ttl=timedelta(milliseconds=150))
    first_lease = await leases.acquire("feishu:chat-1", owner_id="worker-a", now=NOW)
    assert first_lease is not None

    first = repository.claim_job(job_id, lease=first_lease, now=NOW)
    duplicate = repository.claim_job(job_id, lease=first_lease, now=NOW)
    assert duplicate == first
    assert repository.count("runs") == 1
    assert repository.count("run_attempts") == 1

    await asyncio.sleep(0.2)
    second_lease = await leases.acquire(
        "feishu:chat-1",
        owner_id="worker-b",
        now=NOW + timedelta(seconds=2),
    )
    assert second_lease is not None
    recovery = repository.recover_stale_job(
        job_id,
        lease=second_lease,
        now=NOW + timedelta(seconds=2),
        heartbeat_before=NOW + timedelta(seconds=1),
    )
    assert recovery == "recovering"
    second = repository.claim_job(
        job_id,
        lease=second_lease,
        now=NOW + timedelta(seconds=2),
    )

    assert second is not None
    assert second.run_id == first.run_id
    assert second.attempt_no == 2
    assert repository.count("runs") == 1
    assert repository.count("run_attempts") == 2


@pytest.mark.asyncio
async def test_pending_message_is_reclaimed_only_after_lease_and_heartbeat_expire(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish(PublishedJob(job_id, "agent.turn", 0, "feishu:chat-1", "{}"))
    original = await queue.read_next(consumer_id="worker-a")
    assert original is not None

    leases = SessionLeaseManager(redis_client, repository, ttl=timedelta(milliseconds=150))
    first_lease = await leases.acquire("feishu:chat-1", owner_id="worker-a", now=NOW)
    assert first_lease is not None
    assert repository.claim_job(job_id, lease=first_lease, now=NOW) is not None
    reclaimer = PendingMessageReclaimer(repository, queue, leases)

    assert (
        await reclaimer.reclaim_one(
            priority=0,
            consumer_id="worker-b",
            min_idle=timedelta(milliseconds=50),
            heartbeat_before=NOW + timedelta(seconds=1),
        )
        is None
    )
    await asyncio.sleep(0.2)
    reclaimed = await reclaimer.reclaim_one(
        priority=0,
        consumer_id="worker-b",
        min_idle=timedelta(milliseconds=50),
        heartbeat_before=NOW + timedelta(seconds=1),
    )

    assert reclaimed is not None
    assert reclaimed.disposition is PendingDisposition.RESUME
    assert reclaimed.message.message_id == original.message_id
    assert reclaimed.message.job_id == job_id


@pytest.mark.asyncio
async def test_job_reaches_terminal_state_before_stream_ack(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish(PublishedJob(job_id, "agent.turn", 0, "feishu:chat-1", "{}"))
    message = await queue.read_next(consumer_id="worker-a")
    assert message is not None
    leases = SessionLeaseManager(redis_client, repository)
    lease = await leases.acquire("feishu:chat-1", owner_id="worker-a", now=NOW)
    assert lease is not None
    run = repository.claim_job(job_id, lease=lease, now=NOW)
    assert run is not None

    repository.finish_job(run.run_id, lease=lease, outcome="succeeded", now=NOW)
    await queue.acknowledge(message)

    job = repository.get_job(job_id)
    assert job is not None
    assert job.state == "succeeded"
    pending = await redis_client.xpending(queue.stream_key(0), queue.group)
    assert pending["pending"] == 0
    assert not await redis_client.sismember(queue.queued_job_ids_key, job_id)


@pytest.mark.asyncio
async def test_terminal_job_pending_message_is_reclaimed_for_cleanup(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish(PublishedJob(job_id, "agent.turn", 0, "feishu:chat-1", "{}"))
    message = await queue.read_next(consumer_id="worker-dead")
    assert message is not None
    leases = SessionLeaseManager(redis_client, repository)
    lease = await leases.acquire("feishu:chat-1", owner_id="worker-dead", now=NOW)
    assert lease is not None
    run = repository.claim_job(job_id, lease=lease, now=NOW)
    assert run is not None
    repository.finish_job(run.run_id, lease=lease, outcome="succeeded", now=NOW)
    await asyncio.sleep(0.06)

    claimed = await PendingMessageReclaimer(repository, queue, leases).reclaim_one(
        priority=0,
        consumer_id="worker-new",
        min_idle=timedelta(milliseconds=50),
        heartbeat_before=NOW + timedelta(seconds=1),
    )

    assert claimed is not None
    assert claimed.disposition is PendingDisposition.CLEANUP
    await queue.acknowledge(claimed.message)
    pending = await redis_client.xpending(queue.stream_key(0), queue.group)
    assert pending["pending"] == 0
    assert await redis_client.xlen(queue.stream_key(0)) == 0


@pytest.mark.asyncio
async def test_pending_scan_pages_past_ineligible_head_entries(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    for index in range(20):
        await queue.publish(
            PublishedJob(f"missing-job-{index}", "agent.turn", 0, "feishu:other", "{}")
        )
    await queue.publish(PublishedJob(job_id, "agent.turn", 0, "feishu:chat-1", "{}"))
    for _ in range(21):
        assert await queue.read_next(consumer_id="worker-dead") is not None
    await asyncio.sleep(0.02)
    leases = SessionLeaseManager(redis_client, repository)

    claimed = await PendingMessageReclaimer(repository, queue, leases).reclaim_one(
        priority=0,
        consumer_id="worker-new",
        min_idle=timedelta(milliseconds=10),
        heartbeat_before=NOW + timedelta(seconds=1),
    )

    assert claimed is not None
    assert claimed.message.job_id == job_id
    assert claimed.disposition is PendingDisposition.RESUME


@pytest.mark.asyncio
async def test_dead_worker_is_recovered_end_to_end_without_second_run(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish(PublishedJob(job_id, "agent.turn", 0, "feishu:chat-1", "{}"))
    original = await queue.read_next(consumer_id="worker-a")
    assert original is not None
    leases = SessionLeaseManager(redis_client, repository, ttl=timedelta(milliseconds=100))
    first_lease = await leases.acquire("feishu:chat-1", owner_id="worker-a", now=NOW)
    assert first_lease is not None
    first_run = repository.claim_job(job_id, lease=first_lease, now=NOW)
    assert first_run is not None
    await asyncio.sleep(0.12)

    pending_claim = await PendingMessageReclaimer(repository, queue, leases).reclaim_one(
        priority=0,
        consumer_id="worker-b",
        min_idle=timedelta(milliseconds=50),
        heartbeat_before=NOW + timedelta(seconds=1),
    )
    assert pending_claim is not None
    assert pending_claim.disposition is PendingDisposition.RESUME
    second_lease = await leases.acquire(
        "feishu:chat-1",
        owner_id="worker-b",
        now=NOW + timedelta(seconds=2),
    )
    assert second_lease is not None
    assert (
        repository.recover_stale_job(
            job_id,
            lease=second_lease,
            now=NOW + timedelta(seconds=2),
            heartbeat_before=NOW + timedelta(seconds=1),
        )
        == "recovering"
    )
    second_run = repository.claim_job(
        job_id,
        lease=second_lease,
        now=NOW + timedelta(seconds=2),
    )
    assert second_run is not None
    assert second_run.run_id == first_run.run_id
    assert second_run.attempt_no == 2
    repository.finish_job(
        second_run.run_id,
        lease=second_lease,
        outcome="succeeded",
        now=NOW + timedelta(seconds=3),
    )
    await queue.acknowledge(pending_claim.message)

    assert repository.count("runs") == 1
    assert repository.count("run_attempts") == 2
    pending = await redis_client.xpending(queue.stream_key(0), queue.group)
    assert pending["pending"] == 0
    assert await redis_client.xlen(queue.stream_key(0)) == 0


@pytest.mark.asyncio
async def test_uncertain_external_effect_is_not_automatically_retried(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository, job_id = make_repository(tmp_path)
    leases = SessionLeaseManager(redis_client, repository, ttl=timedelta(milliseconds=150))
    first_lease = await leases.acquire("feishu:chat-1", owner_id="worker-a", now=NOW)
    assert first_lease is not None
    run = repository.claim_job(job_id, lease=first_lease, now=NOW)
    assert run is not None
    with connect_database(repository.database) as connection:
        connection.execute(
            """
            INSERT INTO outbound_effects(
                operation_id, run_id, session_key, payload_hash, provider_uuid,
                expected_activity_version, state, owner_id, fencing_epoch,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'unknown', ?, ?, ?, ?)
            """,
            (
                "effect-1",
                run.run_id,
                "feishu:chat-1",
                "hash",
                "provider-uuid",
                1,
                first_lease.owner_id,
                first_lease.epoch,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )

    await asyncio.sleep(0.2)
    second_lease = await leases.acquire(
        "feishu:chat-1",
        owner_id="worker-b",
        now=NOW + timedelta(seconds=2),
    )
    assert second_lease is not None
    recovery = repository.recover_stale_job(
        job_id,
        lease=second_lease,
        now=NOW + timedelta(seconds=2),
        heartbeat_before=NOW + timedelta(seconds=1),
    )

    assert recovery == "needs_review"
    job = repository.get_job(job_id)
    assert job is not None
    assert job.state == "needs_review"
