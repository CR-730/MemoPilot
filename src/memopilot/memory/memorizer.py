"""显式记忆与异步归档共用的记忆生命周期写入器。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from memopilot.memory.contracts import EmbeddingProvider
from memopilot.memory.procedures import build_procedure_rule_schema, build_trigger_tags
from memopilot.memory.store import MemoryStore

if TYPE_CHECKING:
    from memopilot.extensions.events import EventBus


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    item_id: str
    status: str
    actual_kind: str


class ProcedureTagger(Protocol):
    async def tag(
        self,
        summary: str,
        *,
        tool_requirement: str | None,
        steps: list[str],
    ) -> dict[str, object] | None: ...


class MemoryMemorizer:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        *,
        procedure_tagger: ProcedureTagger | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.procedure_tagger = procedure_tagger
        self.event_bus = event_bus

    async def remember(
        self,
        *,
        summary: str,
        memory_kind: str,
        source_ref: str,
        scope_channel: str = "",
        scope_chat_id: str = "",
        extra: dict[str, object] | None = None,
        happened_at: str | None = None,
        emotional_weight: int = 0,
        explicit_supersedes: str | None = None,
        assert_current: Callable[[], None] | None = None,
        fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> MemoryMutationResult:
        guard = assert_current or (lambda: None)
        write_scope = fenced_write or nullcontext
        text = summary.strip()
        kind = memory_kind.strip()
        if not text or kind not in {"event", "profile", "preference", "procedure"}:
            raise ValueError("记忆摘要为空或 memory_kind 无效")

        metadata = dict(extra or {})
        metadata.setdefault("scope_channel", scope_channel)
        metadata.setdefault("scope_chat_id", scope_chat_id)
        if kind == "procedure":
            kind, metadata = _normalize_procedure(
                text,
                metadata,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
            )
            if kind == "procedure" and self.procedure_tagger is not None:
                try:
                    tags = await self.procedure_tagger.tag(
                        text,
                        tool_requirement=optional_text(metadata.get("tool_requirement")),
                        steps=string_list(metadata.get("steps")),
                    )
                except Exception:
                    tags = None
                if tags:
                    metadata["trigger_tags"] = tags

        embedding = await self.embedder.embed(text)
        guard()
        similar = self.store.search_vectors(
            embedding,
            limit=5,
            memory_types=(kind,),
            score_threshold=(
                0.7 if kind == "procedure" else 0.82 if kind == "event" else 0.9
            ),
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )
        if kind == "event":
            confirmed = recent_event_candidates(
                similar,
                happened_at,
                score_min=0.92,
                score_max=float("inf"),
            )
            duplicate = confirmed[0] if confirmed else None
            if duplicate is not None:
                with write_scope():
                    changed = self.store.reinforce_with_source(
                        str(duplicate["item_id"]),
                        source_ref=source_ref,
                        emotional_weight=bounded_int(emotional_weight),
                    )
                result = MemoryMutationResult(
                    str(duplicate["item_id"]),
                    "reinforced" if changed else "unchanged",
                    kind,
                )
                await self._emit_written(
                    result,
                    summary=text,
                    source_ref=source_ref,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                )
                return result

        if kind == "procedure":
            requirement = optional_text(metadata.get("tool_requirement"))
            merge_target = next(
                (
                    item
                    for item in similar
                    if float(str(item.get("semantic_score") or 0.0)) >= 0.7
                    and optional_text(object_dict(item.get("extra_json")).get("tool_requirement"))
                    == requirement
                ),
                None,
            )
            if merge_target is not None and requirement:
                merged_summary = merge_summary(str(merge_target.get("summary") or ""), text)
                # embedding 可能访问外部服务，必须在 fenced DB 写上下文之外完成。
                merged_embedding = await self.embedder.embed(merged_summary)
                guard()
                with write_scope():
                    store_result = self.store.merge_item(
                        str(merge_target["item_id"]),
                        summary=merged_summary,
                        source_ref=source_ref,
                        embedding=merged_embedding,
                        extra=metadata,
                        happened_at=happened_at,
                        emotional_weight=bounded_int(emotional_weight),
                    )
                mutation = MemoryMutationResult(store_result.item_id, "merged", kind)
                await self._emit_written(
                    mutation,
                    summary=merged_summary,
                    source_ref=source_ref,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                )
                return mutation

        supersede_ids = supersede_candidates(
            self.store,
            embedding=embedding,
            kind=kind,
            extra=metadata,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )
        if explicit_supersedes:
            supersede_ids = (*supersede_ids, explicit_supersedes)
        guard()
        with write_scope():
            store_result = self.store.write_item(
                summary=text,
                memory_type=kind,
                source_ref=source_ref,
                embedding=embedding,
                happened_at=happened_at,
                emotional_weight=bounded_int(emotional_weight),
                extra=metadata,
                supersede_ids=tuple(dict.fromkeys(supersede_ids)),
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
            )
        mutation = MemoryMutationResult(
            store_result.item_id,
            store_result.status,
            kind,
        )
        await self._emit_written(
            mutation,
            summary=text,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )
        return mutation

    async def _emit_written(
        self,
        result: MemoryMutationResult,
        *,
        summary: str,
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
    ) -> None:
        if self.event_bus is None:
            return
        from memopilot.memory.events import MemoryWritten

        session_key = (
            f"{scope_channel}:{scope_chat_id}"
            if scope_channel or scope_chat_id
            else ""
        )
        await self.event_bus.fanout(
            MemoryWritten(
                session_key=session_key,
                source_ref=source_ref,
                memory_type=result.actual_kind,
                item_id=result.item_id,
                status=result.status,
                summary=summary,
            )
        )


def _normalize_procedure(
    summary: str,
    extra: dict[str, object],
    *,
    scope_channel: str,
    scope_chat_id: str,
) -> tuple[str, dict[str, object]]:
    steps = string_list(extra.get("steps"))
    requirement = optional_text(extra.get("tool_requirement"))
    if not requirement and not steps:
        return "preference", {
            "scope_channel": scope_channel,
            "scope_chat_id": scope_chat_id,
        }
    schema = build_procedure_rule_schema(
        summary,
        tool_requirement=requirement,
        steps=steps,
        rule_schema=object_dict(extra.get("rule_schema")),
    )
    schema_for_tags: dict[str, object] = dict(schema)
    extra.update(
        {
            "steps": steps,
            "tool_requirement": requirement,
            "rule_schema": schema,
            "trigger_tags": build_trigger_tags(
                summary,
                tool_requirement=requirement,
                rule_schema=schema_for_tags,
            ),
        }
    )
    return "procedure", extra


def optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def bounded_int(value: object) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        number = 0
    return max(0, min(10, number))


def object_dict(value: object) -> dict[str, object]:
    return {str(key): item for key, item in value.items()} if isinstance(value, dict) else {}


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def supersede_candidates(
    store: MemoryStore,
    *,
    embedding: list[float],
    kind: str,
    extra: dict[str, object],
    scope_channel: str = "",
    scope_chat_id: str = "",
) -> tuple[str, ...]:
    if kind not in {"preference", "procedure", "profile"}:
        return ()
    category = str(extra.get("category") or "")
    if kind == "profile" and category not in {"status", "purchase"}:
        return ()
    hits = store.search_vectors(
        embedding,
        limit=5,
        memory_types=(kind,),
        score_threshold=0.9,
        scope_channel=scope_channel,
        scope_chat_id=scope_chat_id,
    )
    if kind == "profile":
        hits = [
            item
            for item in hits
            if isinstance(item.get("extra_json"), dict)
            and object_dict(item.get("extra_json")).get("category") == category
            and float(str(item.get("semantic_score") or 0.0))
            >= (0.92 if bounded_int(item.get("emotional_weight")) >= 7 else 0.9)
        ]
    return tuple(str(item["item_id"]) for item in hits if item.get("item_id"))


def recent_event_candidates(
    hits: list[dict[str, object]],
    happened_at: object,
    *,
    score_min: float,
    score_max: float,
) -> list[dict[str, object]]:
    reference_text = optional_text(happened_at)
    reference = datetime.fromisoformat(reference_text) if reference_text else datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    candidates: list[dict[str, object]] = []
    for item in hits:
        score = float(str(item.get("semantic_score") or 0.0))
        if not score_min <= score < score_max:
            continue
        item_time_text = optional_text(item.get("happened_at")) or optional_text(
            item.get("created_at")
        )
        if not item_time_text:
            continue
        item_time = datetime.fromisoformat(item_time_text)
        if item_time.tzinfo is None:
            item_time = item_time.replace(tzinfo=UTC)
        if abs((reference - item_time).total_seconds()) <= 7 * 86400:
            candidates.append(item)
    return candidates


def merge_summary(old: str, new: str) -> str:
    left, right = old.strip(), new.strip()
    if not left:
        return right
    if not right or right in left:
        return left
    return f"{left}；补充：{right}"


__all__ = [
    "MemoryMemorizer",
    "MemoryMutationResult",
    "ProcedureTagger",
]
