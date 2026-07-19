from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from typing import Any

import pytest

from memopilot.memory.post_response import PostResponseMemoryWorker, _explicitly_memorized_ids


class _Store:
    def __init__(self) -> None:
        self.superseded: tuple[str, ...] = ()

    def mark_superseded_batch(self, item_ids: tuple[str, ...], **kwargs: Any) -> None:
        self.superseded = item_ids


class _Retriever:
    async def retrieve(self, query: str, **kwargs: Any) -> list[dict[str, object]]:
        assert kwargs["memory_types"] == ("procedure", "preference")
        return [
            {"item_id": "old-1", "summary": "旧 Steam 查询流程", "score": 0.91},
            {"item_id": "low", "summary": "无关规则", "score": 0.4},
        ]


class _Model:
    def __init__(self, topics: list[str], selected: list[str]) -> None:
        self.topics = topics
        self.selected = selected

    async def extract_invalidation_topics(self, user_message: str) -> list[str]:
        return self.topics

    async def select_invalidated_ids(
        self, topic: str, candidates: list[dict[str, object]]
    ) -> list[str]:
        return self.selected


@pytest.mark.asyncio
async def test_post_response_only_supersedes_confirmed_high_score_candidate() -> None:
    store = _Store()
    worker = PostResponseMemoryWorker(
        store,  # type: ignore[arg-type]
        _Retriever(),  # type: ignore[arg-type]
        _Model(["Steam 查询流程"], ["old-1", "unknown"]),  # type: ignore[arg-type]
    )

    result = await worker.run(
        user_message="之前那个 Steam 查询流程不对，以后不要再用了",
        session_key="feishu:chat-1",
    )

    assert result == ("old-1",)
    assert store.superseded == ("old-1",)


@pytest.mark.asyncio
async def test_post_response_model_failure_is_contained() -> None:
    class BrokenModel(_Model):
        async def extract_invalidation_topics(self, user_message: str) -> list[str]:
            raise TimeoutError("light model timeout")

    store = _Store()
    worker = PostResponseMemoryWorker(
        store,  # type: ignore[arg-type]
        _Retriever(),  # type: ignore[arg-type]
        BrokenModel([], []),
    )

    result = await worker.run(user_message="普通讨论", session_key="feishu:chat-1")

    assert result == ()
    assert store.superseded == ()


@pytest.mark.asyncio
async def test_post_response_semantic_threshold_cannot_be_bypassed_by_hotness() -> None:
    class _HotButIrrelevantRetriever:
        async def retrieve(self, query: str, **kwargs: Any) -> list[dict[str, object]]:
            return [
                {
                    "item_id": "old-1",
                    "summary": "高频但无关",
                    "score": 0.99,
                    "semantic_score": 0.4,
                }
            ]

    store = _Store()
    worker = PostResponseMemoryWorker(
        store,  # type: ignore[arg-type]
        _HotButIrrelevantRetriever(),  # type: ignore[arg-type]
        _Model(["Steam 查询流程"], ["old-1"]),  # type: ignore[arg-type]
    )

    result = await worker.run(
        user_message="这个流程不对",
        session_key="feishu:chat-1",
    )

    assert result == ()
    assert store.superseded == ()


@pytest.mark.asyncio
async def test_post_response_rechecks_fence_after_model_before_write() -> None:
    store = _Store()
    worker = PostResponseMemoryWorker(
        store,  # type: ignore[arg-type]
        _Retriever(),  # type: ignore[arg-type]
        _Model(["Steam 查询流程"], ["old-1"]),  # type: ignore[arg-type]
    )

    def lost_fence() -> None:
        raise RuntimeError("lost lease")

    with pytest.raises(RuntimeError, match="lost lease"):
        await worker.run(
            user_message="这个流程不要再用了",
            session_key="feishu:chat-1",
            assert_current=lost_fence,
            fenced_write=nullcontext,
        )

    assert store.superseded == ()


@pytest.mark.asyncio
async def test_post_response_never_supersedes_memory_created_in_same_turn() -> None:
    store = _Store()
    worker = PostResponseMemoryWorker(
        store,  # type: ignore[arg-type]
        _Retriever(),  # type: ignore[arg-type]
        _Model(["Steam 查询流程"], ["old-1"]),  # type: ignore[arg-type]
    )

    result = await worker.run(
        user_message="记住这条流程，但旧流程不对",
        session_key="feishu:chat-1",
        protected_ids={"old-1"},
    )

    assert result == ()
    assert store.superseded == ()


def test_explicit_memory_protection_is_reconstructed_from_step_audit() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE steps(run_id TEXT, tool_name TEXT, state TEXT, observation_json TEXT)"
    )
    connection.executemany(
        "INSERT INTO steps VALUES (?, ?, ?, ?)",
        [
            (
                "run-1",
                "memorize",
                "succeeded",
                json.dumps({"ok": True, "result": {"item_id": "mem-new"}}),
            ),
            ("run-1", "memorize", "failed", json.dumps({"result": {"item_id": "bad"}})),
            ("run-2", "memorize", "succeeded", json.dumps({"result": {"item_id": "other"}})),
        ],
    )

    assert _explicitly_memorized_ids(connection, "run-1") == {"mem-new"}
