from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot.memory.store import MemoryStore
from memopilot.memory.vectorization import VectorizationService
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.tasks.operational import LostLeaseError, OperationalRepository


class _Embedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_vectorization_is_resumable_and_reinforces_exact_duplicate(tmp_path: Path) -> None:
    operational = tmp_path / "operational.db"
    memory = tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory, DatabaseKind.MEMORY)
    now = "2026-07-14T12:00:00+00:00"
    output = {
        "artifacts": {},
        "memories": [
            {"kind": "preference", "summary": "用户偏好中文提交。"},
            {"kind": "event", "summary": "阶段四已经完成。"},
        ],
    }
    with connect_database(operational) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 0)",
            ("feishu:chat-1", "feishu", "chat-1", now, now),
        )
        connection.execute(
            "INSERT INTO consolidation_manifests("
            "consolidation_id, session_key, first_message_id, last_message_id, "
            "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
            "committed_at, artifact_states_json, last_error"
            ") VALUES (?, ?, ?, ?, '{}', ?, 'committed', 1, ?, ?, ?, '{}', NULL)",
            ("con-1", "feishu:chat-1", "m1", "m2", json.dumps(output), now, now, now),
        )
    embedder = _Embedder()
    store = MemoryStore(memory, dimension=2, vector_enabled=False)
    service = VectorizationService(operational, store, embedder)

    first = await service.run("con-1")
    second = await service.run("con-1")

    assert (first.created, first.reinforced, first.unchanged) == (2, 0, 0)
    assert (second.created, second.reinforced, second.unchanged) == (0, 0, 2)
    assert embedder.calls == ["用户偏好中文提交。", "阶段四已经完成。"]

    # 同一事实来自新的 consolidation 时，不复制条目，而是强化原条目。
    output["memories"] = [{"kind": "preference", "summary": "用户偏好中文提交。"}]
    with connect_database(operational) as connection:
        connection.execute(
            "INSERT INTO consolidation_manifests("
            "consolidation_id, session_key, first_message_id, last_message_id, "
            "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
            "committed_at, artifact_states_json, last_error"
            ") VALUES (?, ?, ?, ?, '{}', ?, 'committed', 1, ?, ?, ?, '{}', NULL)",
            ("con-2", "feishu:chat-1", "m3", "m4", json.dumps(output), now, now, now),
        )
    reinforced = await service.run("con-2")
    assert reinforced.reinforced == 1
    item = store.search_keywords("中文提交", limit=1)[0]
    assert item["reinforcement_count"] == 2

    # 明确纠正会创建新事实并把旧事实标为 superseded，检索只返回新版本。
    old_item_id = str(item["item_id"])
    output["memories"] = [
        {
            "kind": "preference",
            "summary": "用户现在允许英文提交。",
            "supersedes": old_item_id,
        }
    ]
    with connect_database(operational) as connection:
        connection.execute(
            "INSERT INTO consolidation_manifests("
            "consolidation_id, session_key, first_message_id, last_message_id, "
            "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
            "committed_at, artifact_states_json, last_error"
            ") VALUES (?, ?, ?, ?, '{}', ?, 'committed', 1, ?, ?, ?, '{}', NULL)",
            ("con-3", "feishu:chat-1", "m5", "m6", json.dumps(output), now, now, now),
        )

    corrected = await service.run("con-3")

    assert corrected.created == 1
    assert store.get_item(old_item_id)["status"] == "superseded"  # type: ignore[index]
    assert all(
        str(result["item_id"]) != old_item_id
        for result in store.search_keywords("中文提交", limit=4)
    )
    assert store.search_keywords("英文提交", limit=1)[0]["status"] == "active"


@pytest.mark.asyncio
async def test_vectorization_does_not_write_embedding_after_losing_lease(tmp_path: Path) -> None:
    operational = tmp_path / "operational.db"
    memory = tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory, DatabaseKind.MEMORY)
    now = "2026-07-14T12:00:00+00:00"
    output = {"artifacts": {}, "memories": [{"kind": "event", "summary": "旧任务结果"}]}
    with connect_database(operational) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 0)",
            ("feishu:chat-1", "feishu", "chat-1", now, now),
        )
        connection.execute(
            "INSERT INTO consolidation_manifests("
            "consolidation_id, session_key, first_message_id, last_message_id, "
            "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
            "committed_at, artifact_states_json, last_error"
            ") VALUES (?, ?, ?, ?, '{}', ?, 'committed', 1, ?, ?, ?, '{}', NULL)",
            ("con-lost", "feishu:chat-1", "m1", "m2", json.dumps(output), now, now, now),
        )
    repository = OperationalRepository(operational)
    timestamp = datetime.fromisoformat(now)
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="stale", now=timestamp)
    lease = SimpleNamespace(session_key="feishu:chat-1", owner_id="stale", epoch=epoch)

    class _TakeoverEmbedder:
        async def embed(self, text: str) -> list[float]:
            del text
            repository.allocate_fence("feishu:chat-1", owner_id="current", now=timestamp)
            return [1.0, 0.0]

    store = MemoryStore(memory, dimension=2, vector_enabled=False)
    service = VectorizationService(operational, store, _TakeoverEmbedder())

    with pytest.raises(LostLeaseError, match="失效"):
        await service.run(
            "con-lost",
            assert_current=lambda: repository.assert_current_fence(lease),
            fenced_write=lambda: repository.fenced_write(lease),
        )

    assert store.search_keywords("旧任务", limit=5) == []
    with connect_database(memory) as connection:
        assert (
            connection.execute(
                "SELECT state FROM memory_ingestion_batches WHERE batch_id = 'vectorize:con-lost'"
            ).fetchone()[0]
            == "writing"
        )
