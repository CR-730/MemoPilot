from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
async def test_vector_store_failure_still_returns_keyword_lane() -> None:
    class _BrokenVectorStore(_Store):
        def search_vectors(self, vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("sqlite-vec unavailable")

    store = _BrokenVectorStore()
    hits = await MemoryRetriever(
        store,  # type: ignore[arg-type]
        _Embedder(),
        score_threshold=0.0,
    ).retrieve("原始问题", limit=3)

    assert [item["item_id"] for item in hits] == ["b"]
    assert store.keyword_queries == ["原始问题"]


@pytest.mark.asyncio
async def test_rrf_preserves_relevance_score_for_injection_threshold() -> None:
    retriever = MemoryRetriever(
        _Store(),  # type: ignore[arg-type]
        _Embedder(),
        score_threshold=0.45,
    )

    hits = await retriever.retrieve("原始问题", limit=3)
    preference = next(item for item in hits if item["item_id"] == "b")
    block, injected = retriever.build_injection_block(hits)

    assert preference["score"] == pytest.approx(0.8)
    assert preference["rrf_score"] > 0
    assert "b" in injected
    assert "[b]" in block


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


def test_low_score_procedure_with_required_tool_is_always_forced() -> None:
    retriever = MemoryRetriever(
        _Store(),  # type: ignore[arg-type]
        _Embedder(),
        score_thresholds={"procedure": 0.58},
    )
    item = _item("proc-1", "procedure", 0.2, "发送邮件前必须确认")
    item["extra_json"] = {
        "tool_requirement": "send_email",
        "trigger_tags": {
            "scope": "tool_triggered",
            "tools": ["send_email"],
            "skills": [],
            "keywords": [],
        },
    }

    unrelated, unrelated_ids = retriever.build_injection_block([item])
    matched, matched_ids = retriever.build_injection_block([item])

    assert "强制记忆约束" in unrelated
    assert unrelated_ids == ("proc-1",)
    assert "强制记忆约束" in matched
    assert "必须调用工具：send_email" in matched
    assert matched_ids == ("proc-1",)


def test_absolute_threshold_does_not_add_unapproved_relative_score_floor() -> None:
    retriever = MemoryRetriever(
        _Store(),  # type: ignore[arg-type]
        _Embedder(),
        score_thresholds={"preference": 0.8},
        inject_max_procedure_preference=4,
    )
    items = [
        _item("best", "preference", 0.9, "偏好中文"),
        _item("valid", "preference", 0.81, "偏好简洁"),
    ]

    block, injected = retriever.build_injection_block(items)

    assert injected == ("best", "valid")
    assert "偏好简洁" in block


def test_injection_keeps_prototype_confidence_time_and_evidence_metadata() -> None:
    retriever = MemoryRetriever(
        _Store(),  # type: ignore[arg-type]
        _Embedder(),
        score_thresholds={"preference": 0.45},
    )
    item = _item("pref-1", "preference", 0.5, "用户不确定是否喜欢悬疑游戏")
    item["happened_at"] = "2026-07-01T12:00:00+00:00"
    item["source_ref"] = "consolidation:con-1#implicit"

    block, injected = retriever.build_injection_block([item])

    assert injected == ("pref-1",)
    assert "有印象，不确定" in block
    assert "发生于: 2026-07-01 12:00" in block
    assert "距今约" in block
    assert "证据: consolidation:con-1#implicit" in block
    assert "低置信线索: 不能单独证明历史细节" in block


def test_procedure_injection_includes_steps_and_tool_constraints() -> None:
    retriever = MemoryRetriever(
        _Store(),  # type: ignore[arg-type]
        _Embedder(),
        inject_line_max=240,
    )
    item = _item("proc-1", "procedure", 0.9, "发送邮件前先展示草稿")
    item["extra_json"] = {
        "steps": ["生成草稿", "等待确认", "发送邮件"],
        "rule_schema": {
            "required_tools": ["send_email"],
            "forbidden_tools": ["shell"],
        },
    }

    block, injected = retriever.build_injection_block([item])

    assert injected == ("proc-1",)
    assert "生成草稿 → 等待确认 → 发送邮件" in block
    assert "必须使用：send_email" in block
    assert "禁止使用：shell" in block


def test_high_hotness_cannot_bypass_per_type_semantic_threshold() -> None:
    retriever = MemoryRetriever(
        _Store(),  # type: ignore[arg-type]
        _Embedder(),
        score_thresholds={"preference": 0.52},
    )
    item = _item("pref-low-semantic", "preference", 0.95, "高频但相关性不足")
    item["semantic_score"] = 0.49
    item["hotness"] = 0.99

    block, injected = retriever.build_injection_block([item])

    assert block == ""
    assert injected == ()


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


@pytest.mark.asyncio
async def test_retriever_falls_back_to_keyword_when_embedding_fails() -> None:
    class BrokenEmbedder:
        async def embed(self, text: str) -> list[float]:
            raise RuntimeError("embedding unavailable")

    store = _Store()
    retriever = MemoryRetriever(store, BrokenEmbedder(), score_threshold=0.0)  # type: ignore[arg-type]

    hits = await retriever.retrieve("原始问题", limit=3)

    assert [item["item_id"] for item in hits] == ["b"]
    assert store.vector_queries == []
    assert store.keyword_queries == ["原始问题"]


@pytest.mark.asyncio
async def test_retriever_keeps_successful_queries_when_one_embedding_times_out() -> None:
    class PartialEmbedder:
        async def embed(self, text: str) -> list[float]:
            if text == "超时假设":
                await asyncio.Event().wait()
            return [1.0, 0.0]

    store = _Store()
    retriever = MemoryRetriever(  # type: ignore[arg-type]
        store,
        PartialEmbedder(),
        score_threshold=0.0,
        embed_timeout_seconds=0.01,
    )

    hits = await retriever.retrieve(
        "原始问题",
        aux_queries=("超时假设",),
        limit=3,
    )

    assert [item["item_id"] for item in hits] == ["b", "a"]
    assert len(store.vector_queries) == 1


@pytest.mark.asyncio
async def test_engine_passes_session_scope_to_both_retrieval_lanes() -> None:
    class ScopeStore(_Store):
        def __init__(self) -> None:
            super().__init__()
            self.scopes: list[tuple[str, str]] = []

        def search_vectors(self, vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
            self.scopes.append((kwargs["scope_channel"], kwargs["scope_chat_id"]))
            return super().search_vectors(vector, **kwargs)

        def search_keywords(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            self.scopes.append((kwargs["scope_channel"], kwargs["scope_chat_id"]))
            return super().search_keywords(query, **kwargs)

    store = ScopeStore()
    engine = LayeredMemoryEngine(
        MemoryRetriever(store, _Embedder(), score_threshold=0.0),  # type: ignore[arg-type]
    )

    await engine.query(
        MemoryQuery(
            text="原始问题",
            intent="context",
            session_key="feishu:chat-1",
            limit=2,
        )
    )

    assert store.scopes == [("feishu", "chat-1"), ("feishu", "chat-1")]
