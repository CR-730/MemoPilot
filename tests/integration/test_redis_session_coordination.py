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
async def test_stop_signal_uses_public_stable_key_and_can_be_cleared(
    redis_client: Redis,
) -> None:
    namespace = f"memopilot:test:{uuid4().hex}"
    coordinator = RedisSessionCoordinator(redis_client, namespace=namespace)
    session_key = "feishu:chat-1"

    await coordinator.request_background_stop(session_key, reason="user_message")
    assert await coordinator.background_stop_requested(session_key) is True
    assert await redis_client.exists(coordinator.stop_key(session_key))
    await coordinator.clear_background_stop(session_key)
    assert await coordinator.background_stop_requested(session_key) is False
