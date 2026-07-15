from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.contracts import MemoryQuery
from memopilot.memory.engine import LayeredMemoryEngine
from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.retrieval import MemoryRetriever
from memopilot.memory.store import MemoryStore
from memopilot.memory.vectorization import VectorizationService
from memopilot.memory.worker import MemoryJobRouter
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.engine import TurnInput
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import InboundCommand, OperationalRepository
from memopilot.tasks.outbox import OutboxDispatcher
from memopilot.tasks.redis_queue import RedisTaskQueue
from memopilot.worker.service import WorkerService

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


class _NeverRuntime:
    async def run(self, turn: TurnInput, **kwargs: Any) -> None:
        raise AssertionError("记忆 Job 不应进入 Agent Runtime")


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
    accepted = repository.accept_inbound(
        InboundCommand(
            event_id="event-1",
            message_id="message-1",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "始终使用中文提交"},
            received_at=NOW,
        )
    )
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="seed", now=NOW)
    lease = SimpleNamespace(session_key="feishu:chat-1", owner_id="seed", epoch=epoch)
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content="始终使用中文提交",
        assistant_content="我会记住。",
        now=NOW,
    )

    markdown = MarkdownMemoryStore(tmp_path / "markdown")
    store = MemoryStore(memory_database, dimension=2, vector_enabled=False)
    embedder = _Embedder()
    router = MemoryJobRouter(
        ConsolidationService(
            operational,
            markdown,
            _Extractor(),
            keep_count=0,
            min_new_messages=1,
        ),
        VectorizationService(operational, store, embedder),
        MemoryOptimizer(markdown, _OptimizerModel()),
        repository,
    )
    queue = RedisTaskQueue(memory_redis)
    await queue.ensure_consumer_groups()
    dispatcher = OutboxDispatcher(repository, queue, owner_id="app-memory")
    worker = WorkerService(
        repository,
        queue,
        SessionLeaseManager(memory_redis, repository, ttl=timedelta(seconds=2)),
        RuntimeJobExecutor(
            repository,
            _NeverRuntime(),  # type: ignore[arg-type]
            memory_jobs=router,
        ),
        owner_id="worker-memory",
        clock=lambda: NOW + timedelta(seconds=1),
        heartbeat_interval=0.05,
    )

    # 原 Agent Job 的旧 Outbox 会先被清理，随后真正异步执行 Consolidation。
    assert await dispatcher.dispatch_one(now=NOW) is True
    assert await dispatcher.dispatch_one(now=NOW) is True
    assert await worker.run_once() is True
    assert await worker.run_once() is True
    # Consolidation 提交后才产生 vectorize Outbox。
    assert await dispatcher.dispatch_one(now=datetime.now(UTC) + timedelta(seconds=1)) is True
    assert await worker.run_once() is True

    engine = LayeredMemoryEngine(
        MemoryRetriever(store, embedder, score_threshold=0.0)  # type: ignore[arg-type]
    )
    recalled = await engine.query(
        MemoryQuery(text="Git 提交应该用什么语言", intent="context", limit=4)
    )

    assert "提交信息使用中文" in markdown.read("PENDING.md")
    assert "提交信息使用中文" in recalled.text_block
    assert recalled.records[0].kind == "preference"
