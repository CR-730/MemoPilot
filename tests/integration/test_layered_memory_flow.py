from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.service import MemoryService
from memopilot.memory.store import MemoryStore
from memopilot.memory.vectorization import VectorizationService
from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.agent_task import AgentTask


class _Extractor:
    async def extract(self, conversation: str) -> dict[str, object]:
        assert "中文提交" in conversation
        return {
            "artifacts": {"PENDING.md": "- 中文提交"},
            "memories": [{"kind": "preference", "summary": "中文提交"}],
        }


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _Optimizer:
    async def optimize(self, memory: str, self_text: str, pending: str) -> tuple[str, str]:
        return memory or "# Memory", self_text or "# Self"


@pytest.mark.asyncio
async def test_turn_to_archive_and_vectorization(tmp_path: Path) -> None:
    operational, memory = tmp_path / "operational.db", tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory, DatabaseKind.MEMORY)
    repository = ConversationRepository(operational)
    message = InboundMessage(
        "feishu", "user", "chat-1", "中文提交",
        timestamp=datetime(2026, 7, 14, tzinfo=UTC),
        metadata={"message_id": "message-1"},
    )
    repository.commit_turn(message, assistant_content="记住了")
    markdown = MarkdownMemoryStore(tmp_path / "markdown")
    store = MemoryStore(memory, dimension=2, vector_enabled=False)
    service = MemoryService(
        ConsolidationService(
            operational,
            markdown,
            _Extractor(),
            keep_count=0,
            min_new_messages=1,
            recent_turn_count=1,
        ),
        VectorizationService(operational, store, _Embedder()),
        MemoryOptimizer(markdown, _Optimizer()),
        repository,
    )
    task = AgentTask(
        "consolidate", "memory.consolidate", 3, message.session_key, {}, message.timestamp
    )
    await service.execute_task(task, now=message.timestamp)
    assert "中文提交" in markdown.read("PENDING.md")
    assert store.search_keywords("中文提交", limit=1)[0]["memory_type"] == "preference"
