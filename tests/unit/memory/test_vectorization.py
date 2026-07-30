from __future__ import annotations

import json
from pathlib import Path

import pytest

from memopilot.memory.store import MemoryStore
from memopilot.memory.vectorization import VectorizationService
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database


class _Embedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]


class _ImplicitExtractor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def extract(self, conversation: str) -> list[dict[str, object]]:
        self.calls.append(conversation)
        if self.fail:
            raise RuntimeError("模拟长期记忆提取失败")
        return [
            {
                "kind": "preference",
                "summary": "用户偏好中文提交。",
                "emotional_weight": 0,
                "extra": {},
            }
        ]


@pytest.mark.asyncio
async def test_implicit_extraction_failure_keeps_markdown_commit_and_is_retryable(
    tmp_path: Path,
) -> None:
    operational = tmp_path / "operational.db"
    memory = tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory, DatabaseKind.MEMORY)
    now = "2026-07-14T12:00:00+00:00"
    output = {
        "history_entries": [
            {
                "summary": "[2026-07-14 20:30] 用户完成了阶段四验收。",
                "emotional_weight": 6,
            }
        ],
        "pending_items": [],
        "_conversation": "[2026-07-14T12:00:00+00:00] USER: 我完成阶段四验收了",
        "artifacts": {"HISTORY.md": "[2026-07-14 20:30] 用户完成了阶段四验收。"},
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
            ("con-split", "feishu:chat-1", "m1", "m2", json.dumps(output), now, now, now),
        )
    store = MemoryStore(memory, dimension=2, vector_enabled=False)
    failing = _ImplicitExtractor(fail=True)

    with pytest.raises(RuntimeError, match="长期记忆提取失败"):
        await VectorizationService(
            operational,
            store,
            _Embedder(),
            implicit_extractor=failing,
        ).run("con-split")

    with connect_database(operational) as connection:
        assert connection.execute(
            "SELECT state FROM consolidation_manifests WHERE consolidation_id = 'con-split'"
        ).fetchone()[0] == "committed"
    with connect_database(memory) as connection:
        assert connection.execute(
            "SELECT state FROM memory_ingestion_batches WHERE batch_id = 'vectorize:con-split'"
        ).fetchone()[0] == "failed"

    succeeding = _ImplicitExtractor()
    result = await VectorizationService(
        operational,
        store,
        _Embedder(),
        implicit_extractor=succeeding,
    ).run("con-split")

    assert result.created == 2
    assert succeeding.calls == [output["_conversation"]]
    event = store.search_keywords("阶段四验收", limit=1)[0]
    assert event["memory_type"] == "event"
    assert event["emotional_weight"] == 6
    assert str(event["happened_at"]).startswith("2026-07-14T12:30:00")


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
async def test_vectorization_supersedes_highly_similar_preference_without_explicit_id(
    tmp_path: Path,
) -> None:
    operational = tmp_path / "operational.db"
    memory = tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory, DatabaseKind.MEMORY)
    now = "2026-07-17T12:00:00+00:00"
    with connect_database(operational) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 0)",
            ("feishu:chat-1", "feishu", "chat-1", now, now),
        )
        for consolidation_id, summary in (
            ("con-old", "用户希望每次都优先采用原型实现"),
            ("con-new", "用户现在希望根据实际成本选择实现方式"),
        ):
            output = {
                "artifacts": {},
                "memories": [{"kind": "preference", "summary": summary}],
            }
            connection.execute(
                "INSERT INTO consolidation_manifests("
                "consolidation_id, session_key, first_message_id, last_message_id, "
                "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
                "committed_at, artifact_states_json, last_error"
                ") VALUES (?, ?, ?, ?, '{}', ?, 'committed', 1, ?, ?, ?, '{}', NULL)",
                (
                    consolidation_id,
                    "feishu:chat-1",
                    f"{consolidation_id}-first",
                    f"{consolidation_id}-last",
                    json.dumps(output),
                    now,
                    now,
                    now,
                ),
            )
    store = MemoryStore(memory, dimension=2, vector_enabled=False)
    service = VectorizationService(operational, store, _Embedder())

    await service.run("con-old")
    old = store.search_keywords("原型实现", limit=1)[0]
    await service.run("con-new")

    assert store.get_item(str(old["item_id"]))["status"] == "superseded"  # type: ignore[index]
    assert all(
        str(item["item_id"]) != str(old["item_id"])
        for item in store.search_keywords("原型实现", limit=4)
    )
    assert store.search_keywords("实际成本", limit=1)[0]["status"] == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("emotional_weight", "expected_old_status"),
    [(0, "superseded"), (8, "active")],
)
async def test_vectorization_requires_stronger_evidence_to_supersede_emotional_profile(
    tmp_path: Path,
    emotional_weight: int,
    expected_old_status: str,
) -> None:
    operational = tmp_path / "operational.db"
    memory = tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory, DatabaseKind.MEMORY)
    now = "2026-07-17T12:00:00+00:00"
    old_summary = "用户仍在等待 offer"
    new_summary = "用户开始等待新的 offer"
    with connect_database(operational) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 0)",
            ("feishu:chat-1", "feishu", "chat-1", now, now),
        )
        for consolidation_id, memory_item in (
            (
                "con-old",
                {
                    "kind": "profile",
                    "summary": old_summary,
                    "emotional_weight": emotional_weight,
                    "extra": {"category": "status"},
                },
            ),
            (
                "con-new",
                {
                    "kind": "profile",
                    "summary": new_summary,
                    "extra": {"category": "status"},
                },
            ),
        ):
            output = {"artifacts": {}, "memories": [memory_item]}
            connection.execute(
                "INSERT INTO consolidation_manifests("
                "consolidation_id, session_key, first_message_id, last_message_id, "
                "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
                "committed_at, artifact_states_json, last_error"
                ") VALUES (?, ?, ?, ?, '{}', ?, 'committed', 1, ?, ?, ?, '{}', NULL)",
                (
                    consolidation_id,
                    "feishu:chat-1",
                    f"{consolidation_id}-first",
                    f"{consolidation_id}-last",
                    json.dumps(output),
                    now,
                    now,
                    now,
                ),
            )

    class _ProfileEmbedder:
        async def embed(self, text: str) -> list[float]:
            if text == old_summary:
                return [1.0, 0.0]
            return [0.91, 0.414608]

    store = MemoryStore(memory, dimension=2, vector_enabled=False)
    service = VectorizationService(operational, store, _ProfileEmbedder())

    await service.run("con-old")
    old_item_id = str(store.search_keywords("等待 offer", limit=1)[0]["item_id"])
    await service.run("con-new")

    old = store.get_item(old_item_id)
    assert old is not None
    assert old["status"] == expected_old_status


@pytest.mark.asyncio
async def test_vectorization_merges_same_tool_procedure_and_deduplicates_recent_event(
    tmp_path: Path,
) -> None:
    operational = tmp_path / "operational.db"
    memory = tmp_path / "memory2.db"
    migrate_database(operational, DatabaseKind.OPERATIONAL)
    migrate_database(memory, DatabaseKind.MEMORY)
    now = "2026-07-17T12:00:00+00:00"
    outputs = {
        "procedure-old": {
            "kind": "procedure",
            "summary": "查 Steam 时先用 steam_mcp",
            "extra": {"tool_requirement": "steam_mcp", "steps": ["查询游戏"]},
        },
        "procedure-new": {
            "kind": "procedure",
            "summary": "查 Steam 时先确认区服，再用 steam_mcp",
            "extra": {"tool_requirement": "steam_mcp", "steps": ["确认区服"]},
        },
        "event-old": {
            "kind": "event",
            "summary": "用户完成阶段四检索修复",
            "happened_at": "2026-07-14T12:00:00+00:00",
        },
        "event-new": {
            "kind": "event",
            "summary": "用户完成了阶段四的检索修复",
            "happened_at": "2026-07-17T12:00:00+00:00",
        },
    }
    with connect_database(operational) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 0)",
            ("feishu:chat-1", "feishu", "chat-1", now, now),
        )
        for index, (consolidation_id, item) in enumerate(outputs.items(), 1):
            connection.execute(
                "INSERT INTO consolidation_manifests("
                "consolidation_id, session_key, first_message_id, last_message_id, "
                "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
                "committed_at, artifact_states_json, last_error"
                ") VALUES (?, ?, ?, ?, '{}', ?, 'committed', 1, ?, ?, ?, '{}', NULL)",
                (
                    consolidation_id,
                    "feishu:chat-1",
                    f"m{index}a",
                    f"m{index}b",
                    json.dumps({"artifacts": {}, "memories": [item]}),
                    now,
                    now,
                    now,
                ),
            )
    store = MemoryStore(memory, dimension=2, vector_enabled=False)
    service = VectorizationService(operational, store, _Embedder())

    await service.run("procedure-old")
    procedure_result = await service.run("procedure-new")
    await service.run("event-old")
    event_result = await service.run("event-new")

    assert procedure_result.merged == 1
    procedures = store.search_keywords("Steam", limit=5, memory_types=("procedure",))
    assert len(procedures) == 1
    assert "确认区服" in str(procedures[0]["summary"])
    procedure_extra = procedures[0]["extra_json"]
    assert isinstance(procedure_extra, dict)
    assert procedure_extra["steps"] == ["查询游戏", "确认区服"]
    assert procedure_extra["rule_schema"]["required_tools"] == ["steam_mcp"]
    assert procedure_extra["_merge_note"]
    assert event_result.reinforced == 1
    events = store.search_keywords("阶段四 检索", limit=5, memory_types=("event",))
    assert len(events) == 1
    assert events[0]["reinforcement_count"] == 2
