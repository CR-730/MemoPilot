from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from datetime import UTC, datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path

import pytest

from memopilot.memory import store as memory_store_module
from memopilot.memory.store import MemoryStore
from memopilot.persistence.migrations import DatabaseKind, migrate_database


def _store(tmp_path: Path, *, vector_enabled: bool = True) -> MemoryStore:
    database = tmp_path / ("vector.db" if vector_enabled else "fallback.db")
    migrate_database(database, DatabaseKind.MEMORY)
    return MemoryStore(database, dimension=3, vector_enabled=vector_enabled)


def test_exact_duplicate_reinforces_existing_item_and_records_each_source(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = store.write_item(
        summary="用户喜欢 Python",
        memory_type="preference",
        source_ref="turn:1",
        embedding=[1.0, 0.0, 0.0],
    )
    duplicate = store.write_item(
        summary="用户喜欢  Python ",
        memory_type="preference",
        source_ref="turn:2",
        embedding=[1.0, 0.0, 0.0],
    )

    assert first.status == "created"
    assert duplicate.status == "reinforced"
    assert duplicate.item_id == first.item_id
    item = store.get_item(first.item_id)
    assert item is not None and item["reinforcement_count"] == 2
    assert store.count_sources(first.item_id) == 2


def test_search_scope_keeps_global_memories_but_excludes_other_chat(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_item(
        summary="全局偏好中文回答",
        memory_type="preference",
        source_ref="global",
        embedding=[1.0, 0.0, 0.0],
    )
    store.write_item(
        summary="会话一偏好中文回答",
        memory_type="preference",
        source_ref="chat-1",
        embedding=[1.0, 0.0, 0.0],
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )
    store.write_item(
        summary="会话二偏好中文回答",
        memory_type="preference",
        source_ref="chat-2",
        embedding=[1.0, 0.0, 0.0],
        scope_channel="feishu",
        scope_chat_id="chat-2",
    )

    hits = store.search_keywords(
        "中文回答",
        limit=10,
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )

    assert {item["source_ref"] for item in hits} == {"global", "chat-1"}


def test_scope_filter_cannot_be_starved_by_other_chat_before_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(12):
        store.write_item(
            summary=f"共享检索词 其他会话 {index}",
            memory_type="preference",
            source_ref=f"other:{index}",
            embedding=[1.0, 0.0, 0.0],
            scope_channel="feishu",
            scope_chat_id="other-chat",
        )
    local = store.write_item(
        summary="共享检索词 当前会话",
        memory_type="preference",
        source_ref="local",
        embedding=[0.8, 0.6, 0.0],
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )

    vector_hits = store.search_vectors(
        [1.0, 0.0, 0.0],
        limit=1,
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )
    keyword_hits = store.search_keywords(
        "共享检索词",
        limit=1,
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )

    assert [item["item_id"] for item in vector_hits] == [local.item_id]
    assert [item["item_id"] for item in keyword_hits] == [local.item_id]


def test_timeline_filters_scope_before_limit_and_compares_real_instants(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(4):
        store.write_item(
            summary=f"其他会话较新事件 {index}",
            memory_type="event",
            source_ref=f"other-event:{index}",
            embedding=[1.0, 0.0, 0.0],
            happened_at=f"2026-07-17T0{index + 2}:00:00+00:00",
            scope_channel="feishu",
            scope_chat_id="other-chat",
        )
    local = store.write_item(
        summary="北京时间当天凌晨事件",
        memory_type="event",
        source_ref="local-event",
        embedding=[1.0, 0.0, 0.0],
        happened_at="2026-07-16T16:30:00+00:00",
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )
    china = timezone(timedelta(hours=8))

    hits = store.list_events(
        time_start=datetime(2026, 7, 17, tzinfo=china),
        time_end=datetime(2026, 7, 18, tzinfo=china),
        limit=1,
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )

    assert [item["item_id"] for item in hits] == [local.item_id]


def test_time_filtered_search_excludes_items_without_happened_at(tmp_path: Path) -> None:
    store = _store(tmp_path, vector_enabled=False)
    store.write_item(
        summary="没有事件时间的偏好",
        memory_type="preference",
        source_ref="no-time",
        embedding=[1.0, 0.0, 0.0],
    )

    hits = store.search_vectors(
        [1.0, 0.0, 0.0],
        limit=5,
        time_start=datetime(2026, 7, 17, tzinfo=UTC),
        time_end=datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert hits == []


def test_same_summary_in_different_scopes_remains_independent(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = store.write_item(
        summary="用户希望简洁回答",
        memory_type="preference",
        source_ref="chat-1:same",
        embedding=[1.0, 0.0, 0.0],
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )
    second = store.write_item(
        summary="用户希望简洁回答",
        memory_type="preference",
        source_ref="chat-2:same",
        embedding=[1.0, 0.0, 0.0],
        scope_channel="feishu",
        scope_chat_id="chat-2",
    )

    assert first.item_id != second.item_id
    assert first.status == second.status == "created"


def test_reinforce_items_once_is_idempotent_per_usage_ref(tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = store.write_item(
        summary="用户偏好简洁回复",
        memory_type="preference",
        source_ref="turn:1",
        embedding=[1.0, 0.0, 0.0],
    )

    assert store.reinforce_items_once((item.item_id,), usage_ref="turn:2") == 1
    assert store.reinforce_items_once((item.item_id,), usage_ref="turn:2") == 0

    saved = store.get_item(item.item_id)
    assert saved is not None
    assert saved["reinforcement_count"] == 2


def test_merge_procedure_updates_same_item_and_records_new_source(tmp_path: Path) -> None:
    store = _store(tmp_path, vector_enabled=False)
    created = store.write_item(
        summary="查询 Steam 时先用 steam_mcp",
        memory_type="procedure",
        source_ref="procedure:old",
        embedding=[1.0, 0.0, 0.0],
        extra={"tool_requirement": "steam_mcp", "steps": ["查询游戏"]},
    )

    merged = store.merge_item(
        created.item_id,
        summary="查询 Steam 时先确认区服，再用 steam_mcp",
        source_ref="procedure:new",
        embedding=[0.9, 0.1, 0.0],
        extra={"tool_requirement": "steam_mcp", "steps": ["确认区服", "查询游戏"]},
    )

    assert merged.item_id == created.item_id
    assert merged.status == "merged"
    item = store.get_item(created.item_id)
    assert item is not None
    assert "确认区服" in str(item["summary"])
    assert item["reinforcement_count"] == 2
    assert store.count_sources(created.item_id) == 2


def test_reinforce_with_source_is_idempotent_for_event_redelivery(tmp_path: Path) -> None:
    store = _store(tmp_path, vector_enabled=False)
    created = store.write_item(
        summary="用户完成阶段四",
        memory_type="event",
        source_ref="event:old",
        embedding=[1.0, 0.0, 0.0],
        happened_at="2026-07-14T12:00:00+00:00",
    )

    assert store.reinforce_with_source(created.item_id, source_ref="event:new")
    assert not store.reinforce_with_source(created.item_id, source_ref="event:new")
    item = store.get_item(created.item_id)
    assert item is not None and item["reinforcement_count"] == 2


def test_reconfirmed_exact_memory_reactivates_superseded_item(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.write_item(
        summary="用户偏好中文回复",
        memory_type="preference",
        source_ref="turn:old",
        embedding=[1.0, 0.0, 0.0],
    )
    store.mark_superseded(first.item_id)

    reconfirmed = store.write_item(
        summary="用户偏好中文回复",
        memory_type="preference",
        source_ref="turn:new",
        embedding=[1.0, 0.0, 0.0],
    )

    assert reconfirmed.item_id == first.item_id
    assert reconfirmed.status == "reinforced"
    assert store.get_item(first.item_id)["status"] == "active"  # type: ignore[index]


def test_sqlite_vec_and_cosine_fallback_return_the_same_nearest_item(tmp_path: Path) -> None:
    vector_store = _store(tmp_path, vector_enabled=True)
    fallback_store = _store(tmp_path, vector_enabled=False)
    for store in (vector_store, fallback_store):
        store.write_item(
            summary="Python Agent",
            memory_type="procedure",
            source_ref="source:python",
            embedding=[1.0, 0.0, 0.0],
        )
        store.write_item(
            summary="Redis Queue",
            memory_type="procedure",
            source_ref="source:redis",
            embedding=[0.0, 1.0, 0.0],
        )

    vector_hits = vector_store.search_vectors([0.9, 0.1, 0.0], limit=2, score_threshold=0)
    fallback_hits = fallback_store.search_vectors([0.9, 0.1, 0.0], limit=2, score_threshold=0)

    assert vector_store.vector_available is True
    assert fallback_store.vector_available is False
    assert vector_hits[0]["summary"] == fallback_hits[0]["summary"] == "Python Agent"


def test_hotness_matches_prototype_frequency_and_half_life() -> None:
    now = datetime(2026, 7, 17, tzinfo=UTC)

    fresh = memory_store_module._hotness_score(  # type: ignore[attr-defined]
        1,
        now,
        now=now,
        half_life_days=14,
        emotional_weight=0,
    )
    aged = memory_store_module._hotness_score(  # type: ignore[attr-defined]
        1,
        now - timedelta(days=14),
        now=now,
        half_life_days=14,
        emotional_weight=0,
    )
    emotional = memory_store_module._hotness_score(  # type: ignore[attr-defined]
        1,
        now - timedelta(days=21),
        now=now,
        half_life_days=14,
        emotional_weight=10,
    )

    assert fresh == pytest.approx(2 / 3)
    assert aged == pytest.approx(1 / 3)
    assert emotional == pytest.approx(1 / 3)


def test_semantic_threshold_is_applied_before_hotness_reranking(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_item(
        summary="高热度但语义不足的记忆",
        memory_type="preference",
        source_ref="turn:threshold",
        embedding=[1.0, 0.0, 0.0],
    )
    query = [0.44, math.sqrt(1 - 0.44**2), 0.0]

    hits = store.search_vectors(query, limit=4, score_threshold=0.45)

    assert hits == []


def test_keyword_lane_indexes_cjk_bigrams_and_ascii_terms(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_item(
        summary="用户准备使用 Python 开发记忆 Agent",
        memory_type="event",
        source_ref="turn:keyword",
        embedding=[1.0, 0.0, 0.0],
    )

    chinese = store.search_keywords("记忆系统", limit=5)
    ascii_hits = store.search_keywords("Python", limit=5)

    assert chinese[0]["source_ref"] == "turn:keyword"
    assert ascii_hits[0]["source_ref"] == "turn:keyword"


def test_v1_upgrade_rebuilds_keyword_and_vector_indexes(tmp_path: Path) -> None:
    database = tmp_path / "legacy-memory.db"
    v1_sql = files("memopilot.persistence.schema").joinpath("memory2_v1.sql").read_text("utf-8")
    summary = "用户长期使用 Python 开发 Agent"
    with sqlite3.connect(database) as connection:
        connection.executescript(v1_sql)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            INSERT INTO memory_items(
                item_id, memory_type, summary, content_hash, source_ref, status,
                reinforcement_count, emotional_weight, happened_at, extra_json,
                created_at, updated_at
            ) VALUES ('legacy-1', 'preference', ?, ?, 'legacy:1', 'active', 1, 0,
                      NULL, '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
            """,
            (summary, hashlib.sha256(f"preference\0{summary.casefold()}".encode()).hexdigest()),
        )
        connection.execute(
            """
            INSERT INTO memory_embeddings(
                item_id, provider, model, dimension, embedding, content_hash, created_at, updated_at
            ) VALUES ('legacy-1', 'test', 'test', 3, ?, 'hash',
                      '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
            """,
            (struct.pack("<3f", 1.0, 0.0, 0.0),),
        )

    migrate_database(database, DatabaseKind.MEMORY)
    store = MemoryStore(database, dimension=3, vector_enabled=True)

    assert store.search_keywords("Python", limit=5)[0]["item_id"] == "legacy-1"
    assert store.search_vectors([1.0, 0.0, 0.0], limit=5)[0]["item_id"] == "legacy-1"
