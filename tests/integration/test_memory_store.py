from __future__ import annotations

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
