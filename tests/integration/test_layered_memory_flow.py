from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.bus.events import InboundMessage
from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.contracts import MemoryQuery
from memopilot.memory.engine import LayeredMemoryEngine
from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.retrieval import MemoryRetriever
from memopilot.memory.service import MemoryService
from memopilot.memory.store import MemoryStore
from memopilot.memory.vectorization import VectorizationService
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.agent_loop import AgentLoop
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.redis_queue import RedisTaskQueue
from memopilot.tasks.session_coordination import RedisSessionCoordinator

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def memory_redis() -> AsyncIterator[Redis]:
    url = os.getenv("MEMOPILOT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
    client = Redis.from_url(url, decode_responses=True)
    await client.ping()
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


class _Extractor:
    async def extract(self, conversation: str) -> dict[str, object]:
        assert "始终使用中文提交" in conversation
        return {
            "artifacts": {
                "PENDING.md": "- [preference] 用户要求 Git 提交信息使用中文。",
                "HISTORY.md": "## 2026-07-14\n\n用户明确了提交语言偏好。",
            },
            "memories": [{"kind": "preference", "summary": "用户要求 Git 提交信息使用中文。"}],
        }


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _OptimizerModel:
    async def optimize(self, memory: str, self_text: str, pending: str) -> tuple[str, str]:
        return memory or "# 长期记忆", self_text or "# MemoPilot"


class _PostResponse:
    async def run(
        self,
        *,
        turn_id: str,
        session_key: str,
        **kwargs: Any,
    ) -> tuple[str, ...]:
        return ()


class _MemoryDispatcher:
    def __init__(self, router: MemoryService) -> None:
        self.router = router

    async def dispatch(self, task, *, lease, now) -> tuple[()]:
        del now
        await self.router.execute_task(task, lease=lease, now=NOW)
        return ()


@pytest.mark.asyncio
async def test_turn_to_async_archive_vector_and_next_turn_recall(
    tmp_path: Path,
    memory_redis: Redis,
) -> None:
    operational = tmp_path / "operational.db"
    memory_database = tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory_database, DatabaseKind.MEMORY)
    repository = OperationalRepository(operational)
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "始终使用中文提交",
        timestamp=NOW,
        metadata={"message_id": "message-1"},
    )
    repository.record_inbound_activity(message)
    tasks = repository.commit_turn(
        message,
        assistant_content="我会记住。",
    )

    markdown = MarkdownMemoryStore(tmp_path / "markdown")
    store = MemoryStore(memory_database, dimension=2, vector_enabled=False)
    embedder = _Embedder()
    router = MemoryService(
        ConsolidationService(
            operational,
            markdown,
            _Extractor(),
            keep_count=0,
            min_new_messages=1,
            recent_turn_count=1,
        ),
        VectorizationService(operational, store, embedder),
        MemoryOptimizer(markdown, _OptimizerModel()),
        repository,
        post_response=_PostResponse(),  # type: ignore[arg-type]
    )
    queue = RedisTaskQueue(memory_redis)
    await queue.ensure_consumer_groups()
    assert tasks is not None
    for task in tasks.background_tasks:
        assert await queue.publish_task_once(task) is not None
    runner = AgentLoop(
        queue,
        SessionLeaseManager(memory_redis, repository, ttl=timedelta(seconds=2)),
        _MemoryDispatcher(router),
        owner_id="runner-memory",
        session_coordinator=RedisSessionCoordinator(memory_redis),
        clock=lambda: NOW + timedelta(seconds=1),
        heartbeat_interval=0.05,
    )

    assert await runner.run_once() is True
    assert await runner.run_once() is True

    engine = LayeredMemoryEngine(
        MemoryRetriever(store, embedder, score_threshold=0.0)  # type: ignore[arg-type]
    )
    recalled = await engine.query(
        MemoryQuery(text="Git 提交应该用什么语言", intent="context", limit=4)
    )

    assert "提交信息使用中文" in markdown.read("PENDING.md")
    assert "提交信息使用中文" in recalled.text_block
    assert recalled.records[0].kind == "preference"
