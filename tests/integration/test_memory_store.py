from __future__ import annotations

import hashlib
import sqlite3
import struct
from importlib.resources import files
from pathlib import Path

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
