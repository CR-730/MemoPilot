"""记忆 intent 路由与结构化结果适配。"""

from __future__ import annotations

import asyncio
from typing import Any

from memopilot.memory.contracts import (
    HypothesisProvider,
    MemoryEvidence,
    MemoryQuery,
    MemoryQueryResult,
    MemoryRecord,
)
from memopilot.memory.retrieval import MemoryRetriever


class LayeredMemoryEngine:
    def __init__(
        self,
        retriever: MemoryRetriever,
        *,
        hypothesis_provider: HypothesisProvider | None = None,
    ) -> None:
        self.retriever = retriever
        self.hypothesis_provider = hypothesis_provider

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        limit = max(1, min(request.limit, 200))
        scope_channel, _, scope_chat_id = request.session_key.partition(":")
        if request.intent == "timeline":
            if request.time_start is None or request.time_end is None:
                return MemoryQueryResult(trace={"intent": "timeline", "missing_time": True})
            hits = self.retriever.store.list_events(
                time_start=request.time_start,
                time_end=request.time_end,
                limit=limit,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
            )
            return self._result(request.intent, hits, aux_queries=())

        memory_types = request.memory_kinds
        if not memory_types and request.intent == "interest":
            memory_types = ("preference", "profile")
        elif not memory_types and request.intent == "procedure":
            memory_types = ("procedure", "preference")

        aux_queries: tuple[str, ...] = ()
        if request.intent == "answer" and self.hypothesis_provider is not None:
            generated = await asyncio.gather(
                self.hypothesis_provider.generate(request.text, style="event"),
                self.hypothesis_provider.generate(request.text, style="general"),
                return_exceptions=True,
            )
            aux_queries = tuple(
                value.strip()
                for value in generated
                if isinstance(value, str) and value.strip()
            )
        hits = await self.retriever.retrieve(
            request.text,
            aux_queries=aux_queries,
            memory_types=memory_types,
            limit=limit,
            time_start=request.time_start,
            time_end=request.time_end,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )
        return self._result(
            request.intent,
            hits,
            aux_queries=aux_queries,
        )

    def _result(
        self,
        intent: str,
        hits: list[dict[str, Any]],
        *,
        aux_queries: tuple[str, ...],
    ) -> MemoryQueryResult:
        text_block = ""
        injected_ids: tuple[str, ...] = ()
        if intent in {"context", "procedure"}:
            text_block, injected_ids = self.retriever.build_injection_block(hits)
        records = tuple(
            MemoryRecord(
                id=str(item.get("item_id") or ""),
                kind=str(item.get("memory_type") or ""),
                summary=str(item.get("summary") or ""),
                score=float(item.get("score") or 0.0),
                evidence=(MemoryEvidence(str(item.get("source_ref") or "")),),
                signals={
                    "reinforcement_count": item.get("reinforcement_count", 1),
                    "emotional_weight": item.get("emotional_weight", 0),
                },
                injected=str(item.get("item_id") or "") in injected_ids,
            )
            for item in hits
        )
        if intent == "interest":
            text_block = "\n---\n".join(record.summary for record in records)
        return MemoryQueryResult(
            text_block=text_block,
            records=records,
            trace={
                "intent": intent,
                "aux_queries": list(aux_queries),
                "hit_count": len(records),
                "candidate_ids": [record.id for record in records if record.id],
                "injected_ids": list(injected_ids),
            },
        )


__all__ = ["LayeredMemoryEngine"]
