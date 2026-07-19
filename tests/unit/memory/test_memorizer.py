from __future__ import annotations

from datetime import UTC, datetime

from memopilot.memory.memorizer import MemoryMemorizer
from memopilot.memory.store import MemoryStore
from memopilot.persistence.migrations import DatabaseKind, migrate_database


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _AmbiguousEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if "旧" in text else [0.85, 0.5267826876]


class _ProcedureTagger:
    async def tag(
        self,
        summary: str,
        *,
        tool_requirement: str | None,
        steps: list[str],
    ) -> dict[str, object]:
        return {
            "scope": "tool_triggered",
            "tools": [tool_requirement] if tool_requirement else [],
            "skills": [],
            "keywords": ["邮件"],
        }


def _memorizer(tmp_path) -> MemoryMemorizer:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    return MemoryMemorizer(
        MemoryStore(database, dimension=2, vector_enabled=False),
        _Embedder(),
    )


async def test_procedure_without_execution_condition_downgrades_to_preference(tmp_path) -> None:
    memorizer = _memorizer(tmp_path)

    result = await memorizer.remember(
        summary="用户喜欢简洁回答",
        memory_kind="procedure",
        source_ref="turn:1",
        scope_channel="feishu",
        scope_chat_id="chat-1",
        extra={},
    )

    assert result.actual_kind == "preference"
    assert memorizer.store.get_item(result.item_id)["memory_type"] == "preference"  # type: ignore[index]


async def test_procedure_merge_keeps_stable_id_and_adds_source(tmp_path) -> None:
    memorizer = _memorizer(tmp_path)
    first = await memorizer.remember(
        summary="发送邮件前先展示草稿",
        memory_kind="procedure",
        source_ref="turn:1",
        scope_channel="feishu",
        scope_chat_id="chat-1",
        extra={"tool_requirement": "send_email", "steps": ["展示草稿"]},
    )
    second = await memorizer.remember(
        summary="发送邮件前等待用户确认",
        memory_kind="procedure",
        source_ref="turn:2",
        scope_channel="feishu",
        scope_chat_id="chat-1",
        extra={"tool_requirement": "send_email", "steps": ["等待确认"]},
    )

    assert second.status == "merged"
    assert second.item_id == first.item_id
    assert memorizer.store.count_sources(first.item_id) == 2


async def test_procedure_tagger_makes_natural_chinese_request_reachable(tmp_path) -> None:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    store = MemoryStore(database, dimension=2, vector_enabled=False)
    memorizer = MemoryMemorizer(
        store,
        _Embedder(),
        procedure_tagger=_ProcedureTagger(),
    )
    result = await memorizer.remember(
        summary="发送邮件前先让我确认",
        memory_kind="procedure",
        source_ref="turn:procedure",
        extra={"tool_requirement": "send_email", "steps": ["等待确认"]},
    )
    item = store.get_item(result.item_id)
    assert item is not None
    item["score"] = 0.2

    from memopilot.memory.retrieval import MemoryRetriever

    block, injected = MemoryRetriever(
        store, _Embedder(), score_thresholds={"procedure": 0.58}
    ).build_injection_block([item])

    assert "强制记忆约束" in block
    assert injected == (result.item_id,)


async def test_ambiguous_event_candidate_keeps_new_event_without_extra_llm(tmp_path) -> None:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    store = MemoryStore(database, dimension=2, vector_enabled=False)
    old = store.write_item(
        summary="旧事件：用户完成了阶段四",
        memory_type="event",
        source_ref="turn:old",
        embedding=[1.0, 0.0],
    )
    memorizer = MemoryMemorizer(store, _AmbiguousEmbedder())

    result = await memorizer.remember(
        summary="新描述：阶段四已经完成",
        memory_kind="event",
        source_ref="turn:new",
    )

    assert result.item_id != old.item_id
    assert result.status == "created"


async def test_identical_event_after_seven_day_window_creates_new_instance(
    tmp_path,
) -> None:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    store = MemoryStore(database, dimension=2, vector_enabled=False)
    memorizer = MemoryMemorizer(store, _Embedder())

    first = await memorizer.remember(
        summary="用户完成阶段四",
        memory_kind="event",
        source_ref="turn:day-0",
        happened_at="2026-07-01T10:00:00+00:00",
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )
    second = await memorizer.remember(
        summary="用户完成阶段四",
        memory_kind="event",
        source_ref="turn:day-8",
        happened_at="2026-07-09T10:00:00+00:00",
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )

    assert first.status == "created"
    assert second.status == "created"
    assert second.item_id != first.item_id
    events = store.list_events(
        time_start=datetime(2026, 7, 1, tzinfo=UTC),
        time_end=datetime(2026, 7, 10, tzinfo=UTC),
        limit=10,
        scope_channel="feishu",
        scope_chat_id="chat-1",
    )
    assert {event["item_id"] for event in events} == {first.item_id, second.item_id}
