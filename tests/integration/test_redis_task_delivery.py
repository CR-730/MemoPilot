from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.bus.events import InboundMessage
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.redis_queue import PublishedTask, RedisTaskQueue
from memopilot.tasks.session_coordination import RedisSessionCoordinator

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
async def test_inbound_publish_atomically_creates_p0_and_stop_signal(
    redis_client: Redis,
) -> None:
    queue = RedisTaskQueue(redis_client)
    coordinator = RedisSessionCoordinator(redis_client)
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "你好",
        timestamp=NOW,
        metadata={"message_id": "message-1"},
    )

    first = await queue.publish_inbound(
        message,
        stop_key=coordinator.stop_key(message.session_key),
    )
    second = await queue.publish_inbound(
        message,
        stop_key=coordinator.stop_key(message.session_key),
    )

    assert first is not None
    assert second is None
    assert await redis_client.xlen(queue.stream_key(0)) == 1
    assert await redis_client.scard(queue.queued_task_ids_key) == 1
    assert await coordinator.background_stop_requested(message.session_key) is True
    await coordinator.clear_background_stop(message.session_key)

    assert (
        await queue.publish_inbound(
            message,
            stop_key=coordinator.stop_key(message.session_key),
        )
        is None
    )
    assert await coordinator.background_stop_requested(message.session_key) is False


@pytest.mark.asyncio
async def test_consumer_reads_highest_available_priority_first(redis_client: Redis) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish(PublishedTask("task-p3", "drift.run", 3, "feishu:chat", "{}"))
    await queue.publish(PublishedTask("task-p2", "proactive.tick", 2, "feishu:chat", "{}"))
    await queue.publish(PublishedTask("task-p1", "schedule.run", 1, "feishu:chat", "{}"))

    messages = [await queue.read_next(consumer_id="runner-1") for _ in range(3)]

    assert [message.task_id for message in messages] == ["task-p1", "task-p2", "task-p3"]
    for message in messages:
        await queue.acknowledge(message)
    assert await redis_client.scard(queue.queued_task_ids_key) == 0


@pytest.mark.asyncio
async def test_consumer_recreates_group_deleted_while_service_is_running(
    redis_client: Redis,
) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    await queue.publish(PublishedTask("task-1", "memory.optimize", 3, "system:memory", "{}"))
    await redis_client.xgroup_destroy(queue.stream_key(0), queue.group)

    message = await queue.read_next(consumer_id="runner-1")

    assert message is not None
    assert message.task_id == "task-1"


@pytest.mark.asyncio
async def test_unfinished_stream_entries_are_not_trimmed_by_publish(redis_client: Redis) -> None:
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()

    for index in range(10_100):
        await queue.publish(
            PublishedTask(f"task-{index}", "memory.optimize", 3, "system:memory", "{}")
        )

    first = await redis_client.xrange(queue.stream_key(3), count=1)
    assert first[0][1]["task_id"] == "task-0"
