from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from memopilot.tasks.session_coordination import RedisSessionCoordinator


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = Redis.from_url("redis://127.0.0.1:6379/15", decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_user_turn_preempts_background_work_and_releases_after_reply(
    redis_client: Redis,
) -> None:
    namespace = f"memopilot:test:{uuid4().hex}"
    coordinator = RedisSessionCoordinator(redis_client, namespace=namespace)
    session_key = "feishu:chat-1"

    await coordinator.begin_user_turn(session_key, turn_id="turn-1")
    assert await coordinator.user_turn_active(session_key) is True

    await coordinator.request_background_stop(session_key, reason="user_message")
    assert await coordinator.background_stop_requested(session_key) is True

    await coordinator.end_user_turn(session_key, turn_id="turn-1")
    assert await coordinator.user_turn_active(session_key) is False
    await coordinator.clear_background_stop(session_key)
    assert await coordinator.background_stop_requested(session_key) is False


@pytest.mark.asyncio
async def test_session_busy_covers_user_turn_and_background_lease(
    redis_client: Redis,
) -> None:
    namespace = f"memopilot:test:{uuid4().hex}"
    coordinator = RedisSessionCoordinator(redis_client, namespace=namespace)
    session_key = "feishu:chat-1"

    assert await coordinator.session_busy(session_key) is False
    await coordinator.begin_user_turn(session_key, turn_id="turn-1")
    assert await coordinator.session_busy(session_key) is True
    await coordinator.end_user_turn(session_key, turn_id="turn-1")

    digest = coordinator._digest(session_key)
    await redis_client.set(f"{namespace}:lease:{digest}", "runner|epoch|1")
    assert await coordinator.session_busy(session_key) is True
