from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.bus.events import InboundMessage
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.redis_queue import RedisTaskQueue

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


@pytest.mark.asyncio
async def test_background_task_publish_is_atomic_and_idempotent(redis_client: Redis) -> None:
    queue = RedisTaskQueue(redis_client)
    task = AgentTask(
        "memory:bucket:1",
        "memory.optimize",
        3,
        "system:memory",
        {"bucket": 1},
        NOW,
    )

    first = await queue.publish_task_once(task)
    second = await queue.publish_task_once(task)

    assert first is not None
    assert second is None
    assert await redis_client.xlen(queue.stream_key(3)) == 1


@pytest.mark.asyncio
async def test_inbound_publish_atomically_creates_only_one_p0(
    redis_client: Redis,
) -> None:
    queue = RedisTaskQueue(redis_client)
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "你好",
        timestamp=NOW,
        metadata={"message_id": "message-1"},
    )

    first = await queue.publish_inbound(message)
    second = await queue.publish_inbound(message)

    assert first is not None
    assert second is None
    assert await redis_client.xlen(queue.stream_key(0)) == 1
    assert await queue.publish_inbound(message) is None


@pytest.mark.asyncio
async def test_consumer_reads_highest_available_priority_first(redis_client: Redis) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    for task_id, kind, priority in (
        ("task-p3", "drift.run", 3),
        ("task-p2", "proactive.tick", 2),
        ("task-p1", "schedule.run", 1),
    ):
        await queue.publish_task_once(AgentTask(task_id, kind, priority, "feishu:chat", {}, NOW))

    messages = [await queue.read_next(consumer_id="runner-1") for _ in range(3)]

    assert [message.task_id for message in messages] == ["task-p1", "task-p2", "task-p3"]
    for message in messages:
        await queue.acknowledge(message)


@pytest.mark.asyncio
async def test_pending_read_skips_empty_higher_priority_streams(
    redis_client: Redis,
) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish_task_once(
        AgentTask("task-p3", "memory.optimize", 3, "system:memory", {}, NOW)
    )
    original = await queue.read_next(consumer_id="runner-1")
    assert original is not None
    assert original.task_id == "task-p3"

    replayed = await queue.read_pending(consumer_id="runner-1")

    assert replayed is not None
    assert replayed.task_id == "task-p3"


@pytest.mark.asyncio
async def test_consumer_recreates_group_deleted_while_service_is_running(
    redis_client: Redis,
) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish_task_once(
        AgentTask("task-1", "memory.optimize", 3, "system:memory", {}, NOW)
    )
    await redis_client.xgroup_destroy(queue.stream_key(0), queue.group)

    message = await queue.read_next(consumer_id="runner-1")

    assert message is not None
    assert message.task_id == "task-1"


@pytest.mark.asyncio
async def test_unfinished_stream_entries_are_not_trimmed_by_publish(redis_client: Redis) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()

    for index in range(10_100):
        await queue.publish_task_once(
            AgentTask(f"task-{index}", "memory.optimize", 3, "system:memory", {}, NOW)
        )

    first = await redis_client.xrange(queue.stream_key(3), count=1)
    assert first[0][1]["task_id"] == "task-0"
