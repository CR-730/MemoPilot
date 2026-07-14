from __future__ import annotations

from typing import Any

import pytest

from memopilot.memory.contracts import MemoryQuery
from memopilot.memory.engine import LayeredMemoryEngine
from memopilot.memory.retrieval import MemoryRetriever


class _Embedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return {"原始问题": [1.0, 0.0], "事件假设": [0.9, 0.1], "一般假设": [0.8, 0.2]}[text]


class _Store:
    def __init__(self) -> None:
        self.vector_queries: list[list[float]] = []
        self.keyword_queries: list[str] = []

    def search_vectors(self, vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
        self.vector_queries.append(vector)
        if vector == [1.0, 0.0]:
            return [_item("a", "event", 0.9), _item("b", "preference", 0.8)]
        return [_item("c", "profile", 0.85)]

    def search_keywords(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.keyword_queries.append(query)
        return [_item("b", "preference", 1.0)]

    def list_events(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [_item("timeline", "event", 1.0)]


def _item(item_id: str, kind: str, score: float, summary: str | None = None) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "memory_type": kind,
        "summary": summary or f"summary-{item_id}",
        "score": score,
        "source_ref": f"source:{item_id}",
        "reinforcement_count": 1,
        "emotional_weight": 0,
        "updated_at": "2026-07-14T00:00:00+00:00",
        "extra_json": {},
    }


@pytest.mark.asyncio
async def test_retriever_uses_aux_queries_only_for_vector_lane_and_rrf_for_fusion() -> None:
    store = _Store()
    embedder = _Embedder()
    retriever = MemoryRetriever(store, embedder, score_threshold=0.0)  # type: ignore[arg-type]

    hits = await retriever.retrieve(
        "原始问题",
        aux_queries=("事件假设", "一般假设"),
        limit=3,
    )

    assert embedder.calls == ["原始问题", "事件假设", "一般假设"]
    assert store.keyword_queries == ["原始问题"]
    assert [item["item_id"] for item in hits] == ["b", "a", "c"]
    assert hits[0]["rrf_score"] == pytest.approx(1 / 63 + 0.5 / 61)


def test_injection_applies_sections_thresholds_line_and_total_budget() -> None:
    retriever = MemoryRetriever(
        _Store(),  # type: ignore[arg-type]
        _Embedder(),
        score_threshold=0.45,
        inject_max_chars=260,
        inject_line_max=60,
        inject_max_procedure_preference=1,
        inject_max_event_profile=1,
    )
    items = [
        _item("p1", "procedure", 0.9, "先读取设计文档，再编写实现"),
        _item("p2", "preference", 0.8, "这条应被分段数量裁掉"),
        _item("e1", "event", 0.7, "用户完成了阶段三" + "很长" * 50),
        _item("e2", "profile", 0.6, "这条也应被分段数量裁掉"),
        _item("low", "event", 0.1, "低分不注入"),
    ]

    block, injected = retriever.build_injection_block(items)

    assert injected == ("p1", "e1")
    assert "[p1]" in block and "[e1]" in block
    assert "p2" not in block and "e2" not in block and "low" not in block
    assert all(len(line) <= 60 for line in block.splitlines() if line.startswith("- "))
    assert len(block) <= 260


class _Hypotheses:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def generate(self, query: str, *, style: str) -> str:
        self.calls.append((query, style))
        return "事件假设" if style == "event" else "一般假设"


@pytest.mark.asyncio
async def test_context_query_uses_raw_query_without_hyde_but_answer_uses_two_hypotheses() -> None:
    store = _Store()
    embedder = _Embedder()
    hypotheses = _Hypotheses()
    engine = LayeredMemoryEngine(
        MemoryRetriever(store, embedder, score_threshold=0.0),  # type: ignore[arg-type]
        hypothesis_provider=hypotheses,
    )

    context = await engine.query(MemoryQuery(text="原始问题", intent="context", limit=3))
    assert hypotheses.calls == []
    assert context.trace["aux_queries"] == []
    assert context.text_block

    answer = await engine.query(MemoryQuery(text="原始问题", intent="answer", limit=3))
    assert hypotheses.calls == [("原始问题", "event"), ("原始问题", "general")]
    assert answer.trace["aux_queries"] == ["事件假设", "一般假设"]
    assert [record.id for record in answer.records] == ["b", "a", "c"]


@pytest.mark.asyncio
async def test_answer_falls_back_to_raw_query_when_hypothesis_generation_fails() -> None:
    class BrokenHypotheses:
        async def generate(self, query: str, *, style: str) -> str:
            raise TimeoutError(style)

    store = _Store()
    embedder = _Embedder()
    engine = LayeredMemoryEngine(
        MemoryRetriever(store, embedder, score_threshold=0.0),  # type: ignore[arg-type]
        hypothesis_provider=BrokenHypotheses(),
    )

    result = await engine.query(MemoryQuery(text="原始问题", intent="answer", limit=2))

    assert embedder.calls == ["原始问题"]
    assert result.trace["aux_queries"] == []
    assert len(result.records) == 2
