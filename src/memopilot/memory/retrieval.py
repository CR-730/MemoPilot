"""原型语义一致的 Vector + Keyword + RRF 统一检索器。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from memopilot.memory.contracts import EmbeddingProvider


class RetrievalStore(Protocol):
    def search_vectors(self, vector: list[float], **kwargs: Any) -> list[dict[str, Any]]: ...

    def search_keywords(self, query: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    def list_events(self, **kwargs: Any) -> list[dict[str, Any]]: ...


class MemoryRetriever:
    def __init__(
        self,
        store: RetrievalStore,
        embedder: EmbeddingProvider,
        *,
        score_threshold: float = 0.45,
        score_thresholds: dict[str, float] | None = None,
        relative_delta: float = 0.06,
        inject_max_chars: int = 1200,
        inject_line_max: int = 180,
        inject_max_forced: int = 3,
        inject_max_procedure_preference: int = 4,
        inject_max_event_profile: int = 2,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.score_threshold = score_threshold
        self.score_thresholds = score_thresholds or {}
        self.relative_delta = max(0.0, relative_delta)
        self.inject_max_chars = max(120, inject_max_chars)
        self.inject_line_max = max(30, inject_line_max)
        self.inject_max_forced = max(1, inject_max_forced)
        self.inject_max_procedure_preference = max(1, inject_max_procedure_preference)
        self.inject_max_event_profile = max(0, inject_max_event_profile)

    async def retrieve(
        self,
        query: str,
        *,
        aux_queries: Sequence[str] = (),
        memory_types: tuple[str, ...] = (),
        limit: int = 8,
        score_threshold: float | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        keyword_enabled: bool = True,
    ) -> list[dict[str, Any]]:
        queries = tuple(
            dict.fromkeys(text.strip() for text in (query, *aux_queries) if text.strip())
        )
        vectors = await asyncio.gather(*(self.embedder.embed(text) for text in queries))
        vector_by_id: dict[str, dict[str, Any]] = {}
        for vector in vectors:
            hits = self.store.search_vectors(
                vector,
                limit=max(limit, 8),
                memory_types=memory_types,
                score_threshold=(
                    self.score_threshold if score_threshold is None else score_threshold
                ),
                time_start=time_start,
                time_end=time_end,
            )
            for hit in hits:
                item_id = str(hit.get("item_id") or "")
                if not item_id:
                    continue
                previous = vector_by_id.get(item_id)
                if previous is None or _score(hit) > _score(previous):
                    vector_by_id[item_id] = dict(hit)
        vector_hits = sorted(vector_by_id.values(), key=_score, reverse=True)
        keyword_hits = (
            self.store.search_keywords(
                query,
                limit=max(limit * 2, 16),
                memory_types=memory_types,
                time_start=time_start,
                time_end=time_end,
            )
            if keyword_enabled
            else []
        )
        return _rrf_merge(vector_hits, keyword_hits, limit=limit)

    def build_injection_block(
        self, items: Sequence[dict[str, Any]]
    ) -> tuple[str, tuple[str, ...]]:
        sorted_items = sorted(items, key=_score, reverse=True)
        group_best: dict[str, float] = {}
        for item in sorted_items:
            group = _section_group(str(item.get("memory_type") or ""))
            group_best[group] = max(group_best.get(group, 0.0), _score(item))
        sections: dict[str, list[tuple[str, str]]] = {
            "forced": [],
            "rules": [],
            "history": [],
        }
        counts = {"forced": 0, "rules": 0, "history": 0}
        for item in sorted_items:
            kind = str(item.get("memory_type") or "")
            item_id = str(item.get("item_id") or "")
            summary = str(item.get("summary") or "").strip()
            extra = item.get("extra_json") if isinstance(item.get("extra_json"), dict) else {}
            assert isinstance(extra, dict)
            forced = kind == "procedure" and bool(extra.get("tool_requirement"))
            section = "forced" if forced else _section_group(kind)
            if section not in sections or not item_id or not summary:
                continue
            threshold = self.score_thresholds.get(kind, self.score_threshold)
            relative_floor = group_best.get(section, 0.0) - self.relative_delta
            if not forced and _score(item) < max(threshold, relative_floor):
                continue
            cap = {
                "forced": self.inject_max_forced,
                "rules": self.inject_max_procedure_preference,
                "history": self.inject_max_event_profile,
            }[section]
            if counts[section] >= cap:
                continue
            counts[section] += 1
            happened = f"[{item.get('happened_at')}] " if item.get("happened_at") else ""
            suffix = (
                f"（必须调用工具：{extra['tool_requirement']}）"
                if forced
                else ""
            )
            line = _truncate(f"- [{item_id}] {happened}{summary}{suffix}", self.inject_line_max)
            sections[section].append((item_id, line))

        rendered: list[str] = []
        injected: list[str] = []
        for section, title in (
            ("forced", "## 强制记忆约束"),
            ("rules", "## 用户偏好与流程"),
            ("history", "## 相关历史"),
        ):
            entries = sections[section]
            if not entries:
                continue
            part = title + "\n" + "\n".join(line for _, line in entries)
            candidate = "\n\n".join((*rendered, part))
            if len(candidate) > self.inject_max_chars and section != "forced":
                continue
            if len(candidate) > self.inject_max_chars:
                part = _truncate(part, self.inject_max_chars)
                candidate = part
            rendered.append(part)
            injected.extend(item_id for item_id, _ in entries)
        return "\n\n".join(rendered), tuple(injected)


def _rrf_merge(
    vector_items: Sequence[dict[str, Any]],
    keyword_items: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    vector_rank = {str(item["item_id"]): rank for rank, item in enumerate(vector_items, 1)}
    keyword_rank = {str(item["item_id"]): rank for rank, item in enumerate(keyword_items, 1)}
    items = {str(item["item_id"]): dict(item) for item in (*vector_items, *keyword_items)}
    for item_id, item in items.items():
        score = 0.0
        if item_id in vector_rank:
            score += 1.0 / (60 + vector_rank[item_id])
        if item_id in keyword_rank:
            score += 0.5 / (60 + keyword_rank[item_id])
        item["rrf_score"] = score
        item["score"] = score
    return sorted(items.values(), key=lambda item: float(item["rrf_score"]), reverse=True)[:limit]


def _score(item: dict[str, Any]) -> float:
    value = item.get("score", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _section_group(kind: str) -> str:
    if kind in {"procedure", "preference"}:
        return "rules"
    if kind in {"event", "profile"}:
        return "history"
    return "unknown"


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"


__all__ = ["MemoryRetriever"]
