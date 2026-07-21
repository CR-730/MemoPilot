"""原型语义一致的 Vector + Keyword + RRF 统一检索器。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from memopilot.memory.contracts import EmbeddingProvider

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_PHRASES = (
    "未在对话中明确记录",
    "无法凭记忆确认",
    "没有记录",
    "真的没有",
    "未找到",
    "不确定",
)


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
        embed_timeout_seconds: float = 5.0,
        procedure_guard_enabled: bool = True,
        high_inject_delta: float = 0.15,
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
        self.embed_timeout_seconds = max(0.01, embed_timeout_seconds)
        self.procedure_guard_enabled = procedure_guard_enabled
        self.high_inject_delta = max(0.0, high_inject_delta)
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
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> list[dict[str, Any]]:
        queries = tuple(
            dict.fromkeys(text.strip() for text in (query, *aux_queries) if text.strip())
        )
        vector_results = await asyncio.gather(
            *(
                asyncio.wait_for(
                    self.embedder.embed(text),
                    timeout=self.embed_timeout_seconds,
                )
                for text in queries
            ),
            return_exceptions=True,
        )
        vector_by_id: dict[str, dict[str, Any]] = {}
        for vector in vector_results:
            if isinstance(vector, BaseException):
                continue
            try:
                hits = self.store.search_vectors(
                    vector,
                    limit=max(limit, 8),
                    memory_types=memory_types,
                    score_threshold=(
                        self.score_threshold if score_threshold is None else score_threshold
                    ),
                    time_start=time_start,
                    time_end=time_end,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                )
            except Exception as exc:
                logger.warning("向量记忆检索失败，继续使用其他检索 Lane: %s", exc)
                continue
            for hit in hits:
                item_id = str(hit.get("item_id") or "")
                if not item_id:
                    continue
                previous = vector_by_id.get(item_id)
                if previous is None or _score(hit) > _score(previous):
                    vector_by_id[item_id] = dict(hit)
        vector_hits = sorted(vector_by_id.values(), key=_score, reverse=True)
        keyword_hits: list[dict[str, Any]] = []
        if keyword_enabled:
            try:
                keyword_hits = self.store.search_keywords(
                    query,
                    limit=max(limit * 2, 16),
                    memory_types=memory_types,
                    time_start=time_start,
                    time_end=time_end,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                )
            except Exception as exc:
                logger.warning("关键词记忆检索失败，继续使用其他检索 Lane: %s", exc)
        return _rrf_merge(vector_hits, keyword_hits, limit=limit)

    def build_injection_block(
        self,
        items: Sequence[dict[str, Any]],
    ) -> tuple[str, tuple[str, ...]]:
        sorted_items = sorted(items, key=_score, reverse=True)
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
            forced = (
                self.procedure_guard_enabled
                and kind == "procedure"
                and bool(extra.get("tool_requirement"))
            )
            section = "forced" if forced else _section_group(kind)
            if section not in sections or not item_id or not summary:
                continue
            threshold = self.score_thresholds.get(kind, self.score_threshold)
            if not forced and _relevance_score(item) < threshold:
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
            confidence = (
                "有印象，不确定"
                if not forced and _relevance_score(item) < threshold + self.high_inject_delta
                else ""
            )
            suffix = (
                f"（必须调用工具：{extra['tool_requirement']}）"
                if forced
                else ""
            )
            details: list[str] = []
            if kind == "procedure":
                steps = _string_list(extra.get("steps"))
                schema = extra.get("rule_schema")
                rules = schema if isinstance(schema, dict) else {}
                required = _string_list(rules.get("required_tools"))
                forbidden = _string_list(rules.get("forbidden_tools"))
                if steps:
                    details.append("步骤：" + " → ".join(steps))
                if required:
                    details.append("必须使用：" + "、".join(required))
                if forbidden:
                    details.append("禁止使用：" + "、".join(forbidden))
            detail_suffix = "；" + "；".join(details) if details else ""
            meta = _format_memory_meta(item, kind, confidence_label=confidence)
            line = _truncate(
                f"- [{item_id}] {happened}{summary}{suffix}{meta}{detail_suffix}",
                self.inject_line_max,
            )
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
        block = "\n\n".join(rendered)
        visible_ids = tuple(item_id for item_id in injected if f"[{item_id}]" in block)
        return block, visible_ids


def _format_memory_meta(
    item: dict[str, Any],
    memory_type: str,
    *,
    confidence_label: str,
) -> str:
    parts: list[str] = []
    if confidence_label:
        parts.append(confidence_label)
    happened_at = _normalize_happened_at(item.get("happened_at"))
    if happened_at:
        parts.append(f"发生于: {happened_at}")
        age = _format_relative_age(item.get("happened_at"))
        if age:
            parts.append(age)
    source_ref = str(item.get("source_ref") or "").strip()
    parts.append(f"证据: {source_ref}" if source_ref else "证据: 记忆摘要")
    if memory_type == "preference" and any(
        phrase in str(item.get("summary") or "") for phrase in _LOW_CONFIDENCE_PHRASES
    ):
        parts.append("低置信线索: 不能单独证明历史细节")
    return "（" + "；".join(parts) + "）"


def _normalize_happened_at(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 and "T" not in text:
        return parsed.strftime("%Y-%m-%d")
    return parsed.strftime("%Y-%m-%d %H:%M")


def _format_relative_age(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        happened = datetime.fromisoformat(text)
    except ValueError:
        return ""
    delta = datetime.now(happened.tzinfo) - happened
    if delta.days >= 1:
        return f"距今约 {delta.days} 天"
    hours = max(0, int(delta.total_seconds() // 3600))
    if hours >= 1:
        return f"距今约 {hours} 小时"
    return f"距今约 {max(0, int(delta.total_seconds() // 60))} 分钟"


def _rrf_merge(
    vector_items: Sequence[dict[str, Any]],
    keyword_items: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    vector_rank = {str(item["item_id"]): rank for rank, item in enumerate(vector_items, 1)}
    keyword_rank = {str(item["item_id"]): rank for rank, item in enumerate(keyword_items, 1)}
    items = {str(item["item_id"]): dict(item) for item in keyword_items}
    items.update({str(item["item_id"]): dict(item) for item in vector_items})
    for item_id, item in items.items():
        score = 0.0
        if item_id in vector_rank:
            score += 1.0 / (60 + vector_rank[item_id])
        if item_id in keyword_rank:
            score += 0.5 / (60 + keyword_rank[item_id])
        item["rrf_score"] = score
    return sorted(items.values(), key=lambda item: float(item["rrf_score"]), reverse=True)[:limit]


def _score(item: dict[str, Any]) -> float:
    value = item.get("score", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _relevance_score(item: dict[str, Any]) -> float:
    value = item.get("semantic_score", item.get("score", 0.0))
    return float(value) if isinstance(value, int | float) else 0.0


def _section_group(kind: str) -> str:
    if kind in {"procedure", "preference"}:
        return "rules"
    if kind in {"event", "profile"}:
        return "history"
    return "unknown"


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


__all__ = ["MemoryRetriever"]
