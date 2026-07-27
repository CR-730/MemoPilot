"""Turn 完成后的显式否定与纠错处理。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Protocol

from memopilot.memory.retrieval import MemoryRetriever
from memopilot.memory.store import MemoryStore
from memopilot.persistence.migrations import connect_database


class PostResponseModel(Protocol):
    async def extract_invalidation_topics(self, user_message: str) -> list[str]: ...

    async def select_invalidated_ids(
        self,
        topic: str,
        candidates: list[dict[str, object]],
    ) -> list[str]: ...


class PostResponseMemoryWorker:
    def __init__(
        self,
        store: MemoryStore,
        retriever: MemoryRetriever,
        model: PostResponseModel,
        *,
        score_threshold: float = 0.82,
        candidate_limit: int = 5,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.model = model
        self.score_threshold = score_threshold
        self.candidate_limit = candidate_limit

    async def run(
        self,
        *,
        user_message: str,
        session_key: str,
        protected_ids: set[str] | None = None,
        assert_current: Callable[[], None] | None = None,
        fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> tuple[str, ...]:
        guard = assert_current or (lambda: None)
        write_scope = fenced_write or nullcontext
        protected = protected_ids or set()
        try:
            topics = await self.model.extract_invalidation_topics(user_message)
        except Exception:
            return ()
        scope_channel, _, scope_chat_id = session_key.partition(":")
        selected: list[str] = []
        # 每轮最多 1 次主题抽取 + 9 次候选判断，按每次 96 token 计仍低于 1000。
        for topic in topics[:9]:
            if not topic.strip():
                continue
            try:
                hits = await self.retriever.retrieve(
                    topic,
                    memory_types=("procedure", "preference"),
                    limit=self.candidate_limit,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                )
                candidates = [
                    dict(item)
                    for item in hits
                    if float(item.get("semantic_score", item.get("score", 0.0)))
                    >= self.score_threshold
                ][: self.candidate_limit]
                if not candidates:
                    continue
                allowed = {
                    str(item.get("item_id") or "")
                    for item in candidates
                    if str(item.get("item_id") or "") not in protected
                }
                decisions = await self.model.select_invalidated_ids(topic, candidates)
                selected.extend(item_id for item_id in decisions if item_id in allowed)
            except Exception:
                continue
        result = tuple(dict.fromkeys(selected))
        if result:
            guard()
            with write_scope():
                self.store.mark_superseded_batch(
                    result,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                )
        return result


class OperationalPostResponseService:
    def __init__(self, database: Path, worker: PostResponseMemoryWorker) -> None:
        self.database = database
        self.worker = worker

    async def run(
        self,
        *,
        turn_id: str,
        session_key: str,
        protected_ids: set[str] | None = None,
        assert_current: Callable[[], None] | None = None,
        fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> tuple[str, ...]:
        with connect_database(self.database) as connection:
            row = connection.execute(
                "SELECT content FROM messages WHERE turn_id = ? AND role = 'user' "
                "ORDER BY turn_position LIMIT 1",
                (turn_id,),
            ).fetchone()
        if row is None:
            return ()
        return await self.worker.run(
            user_message=str(row["content"]),
            session_key=session_key,
            protected_ids=protected_ids,
            assert_current=assert_current,
            fenced_write=fenced_write,
        )
__all__ = ["OperationalPostResponseService", "PostResponseMemoryWorker", "PostResponseModel"]
