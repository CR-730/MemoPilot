from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from redis.asyncio import Redis

from memopilot.bus.events import InboundMessage
from memopilot.extensions.events import EventBus
from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.agent_core import AgentCore
from memopilot.runtime.agent_loop import AgentLoop
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.engine import DefaultReasoner
from memopilot.runtime.outbound import OutboundDispatch
from memopilot.runtime.passive_turn import PassiveTurnPipeline
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.react import ReActResult
from memopilot.runtime.task_dispatcher import TaskDispatcher
from memopilot.runtime.tools import ToolRegistry
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.redis_queue import RedisTaskQueue

NOW = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)


class _Provider(ChatProvider):
    async def complete(
        self, *, messages: tuple[ChatMessage, ...], tools: tuple[ToolSchema, ...]
    ) -> ModelResponse:
        del messages, tools
        return ModelResponse(content="收到", tool_calls=(), finish_reason="stop")


class _Runtime(DefaultReasoner):
    def __init__(self) -> None:
        super().__init__(_Provider(), ToolRegistry())
        self.calls = 0

    async def run_reasoning(self, *args: object, **kwargs: object) -> ReActResult:
        self.calls += 1
        return await super().run_reasoning(*args, **kwargs)  # type: ignore[arg-type]


class _Outbound:
    def __init__(self) -> None:
        self.sent: list[OutboundDispatch] = []

    async def dispatch(self, dispatch: OutboundDispatch) -> bool:
        self.sent.append(dispatch)
        return True


class _UnusedHandler:
    async def execute_task(
        self,
        task: AgentTask,
        *,
        now: datetime,
    ) -> tuple[AgentTask, ...]:
        raise AssertionError(f"不应路由到 {task.kind}: {now}")


@pytest.mark.asyncio
async def test_real_redis_to_sqlite_passive_turn_and_derived_tasks(
    tmp_path: Path,
) -> None:
    redis_url = os.getenv("MEMOPILOT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
    redis = Redis.from_url(redis_url, decode_responses=True)
    await redis.ping()
    await redis.flushdb()
    event_bus = EventBus()
    try:
        database = tmp_path / "operational.db"
        migrate_database(database, DatabaseKind.OPERATIONAL)
        repository = ConversationRepository(database)
        runtime = _Runtime()
        outbound = _Outbound()
        queue = RedisTaskQueue(redis, namespace="memopilot-e2e")
        await queue.ensure_consumer_groups()
        passive = PassiveTurnPipeline(
            runtime,  # type: ignore[arg-type]
            repository=repository,
            outbound=outbound,
            event_bus=event_bus,
            history_limit=20,
        )
        unused = _UnusedHandler()
        dispatcher = TaskDispatcher(
            passive=AgentCore(passive),
            memory=unused,
            proactive=unused,
            scheduler=unused,
        )
        loop = AgentLoop(queue, dispatcher, clock=lambda: NOW)
        message = InboundMessage(
            "feishu",
            "user-1",
            "chat-1",
            "你好",
            timestamp=NOW,
            metadata={"message_id": "message-1"},
        )
        repository.record_inbound_activity(message)

        assert await queue.publish_inbound(message) is not None
        assert await loop.run_once() is True

        assert runtime.calls == 1
        assert [item.content for item in outbound.sent] == ["收到"]
        history = repository.list_recent_messages(message.session_key, limit=2)
        assert [item.content for item in history] == [
            "你好",
            "收到",
        ]
        assert await redis.xlen(queue.stream_key(0)) == 0
        assert await redis.xlen(queue.stream_key(3)) == 2

        derived = [await queue.read_next(consumer_id="derived-check") for _ in range(2)]
        assert {item.kind for item in derived if item is not None} == {
            "memory.consolidate",
            "memory.post_response",
        }
        for item in derived:
            assert item is not None
            await queue.acknowledge(item)
        assert await redis.xlen(queue.stream_key(3)) == 0
        assert await queue.publish_inbound(message) is None
    finally:
        await event_bus.aclose()
        await redis.flushdb()
        await redis.aclose()
