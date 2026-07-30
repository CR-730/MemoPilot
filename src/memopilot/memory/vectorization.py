"""消费已提交的 Consolidation，并将记忆写入 memory2.db。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from memopilot.memory.contracts import EmbeddingProvider
from memopilot.memory.memorizer import (
    MemoryMemorizer,
    bounded_int,
    object_dict,
    optional_text,
)
from memopilot.memory.store import MemoryStore
from memopilot.persistence.migrations import connect_database


@dataclass(frozen=True, slots=True)
class VectorizationResult:
    created: int = 0
    reinforced: int = 0
    unchanged: int = 0
    merged: int = 0


class ImplicitMemoryExtractor(Protocol):
    async def extract(self, conversation: str) -> list[dict[str, object]]: ...


class VectorizationService:
    def __init__(
        self,
        operational_database: Path,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        *,
        memorizer: MemoryMemorizer | None = None,
        implicit_extractor: ImplicitMemoryExtractor | None = None,
        display_timezone: str = "Asia/Shanghai",
    ) -> None:
        self.operational_database = operational_database
        self.store = store
        self.embedder = embedder
        self.memorizer = memorizer or MemoryMemorizer(store, embedder)
        self.implicit_extractor = implicit_extractor
        self.display_timezone = ZoneInfo(display_timezone)

    async def run(
        self,
        consolidation_id: str,
        *,
        assert_current: Callable[[], None] | None = None,
    ) -> VectorizationResult:
        guard = assert_current or (lambda: None)
        guard()
        with connect_database(self.operational_database) as connection:
            manifest = connection.execute(
                "SELECT m.model_output_json, m.state, s.channel, s.chat_id "
                "FROM consolidation_manifests AS m "
                "JOIN sessions AS s ON s.session_key = m.session_key "
                "WHERE m.consolidation_id = ?",
                (consolidation_id,),
            ).fetchone()
        if manifest is None:
            raise KeyError(consolidation_id)
        if str(manifest["state"]) != "committed":
            raise RuntimeError("找不到指定的 Consolidation Manifest")
        output = json.loads(str(manifest["model_output_json"]))
        if not isinstance(output, dict):
            raise ValueError("Consolidation 尚未提交")
        history_entries = output.get("history_entries", [])
        entries = _event_entries(history_entries, display_timezone=self.display_timezone)
        implicit = output.get("_implicit_memories")
        if isinstance(implicit, list):
            entries.extend(entry for entry in implicit if isinstance(entry, dict))
        elif not entries:
            memories = output.get("memories", [])
            if isinstance(memories, list):
                entries.extend(entry for entry in memories if isinstance(entry, dict))
        batch_id = f"vectorize:{consolidation_id}"
        with connect_database(self.store.database) as connection:
            existing = connection.execute(
                "SELECT state FROM memory_ingestion_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if existing is not None and str(existing["state"]) == "committed":
                return VectorizationResult(unchanged=len(entries))
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT OR IGNORE INTO memory_ingestion_batches("
                "batch_id, source_ref, state, attempts, created_at, updated_at"
                ") VALUES (?, ?, 'pending', 0, ?, ?)",
                (batch_id, f"consolidation:{consolidation_id}", now, now),
            )
            connection.execute(
                "UPDATE memory_ingestion_batches SET state = 'writing', attempts = attempts + 1, "
                "updated_at = ?, last_error = NULL WHERE batch_id = ?",
                (now, batch_id),
            )
        counts = {"created": 0, "reinforced": 0, "unchanged": 0, "merged": 0}
        scope_channel = str(manifest["channel"] or "")
        scope_chat_id = str(manifest["chat_id"] or "")
        try:
            if (
                self.implicit_extractor is not None
                and not isinstance(implicit, list)
                and str(output.get("_conversation") or "").strip()
            ):
                extracted = await self.implicit_extractor.extract(str(output["_conversation"]))
                implicit_entries = [entry for entry in extracted if isinstance(entry, dict)]
                guard()
                output["_implicit_memories"] = implicit_entries
                connection = connect_database(self.operational_database)
                connection.execute("BEGIN IMMEDIATE")
                try:
                    cursor = connection.execute(
                        "UPDATE consolidation_manifests SET model_output_json = ?, "
                        "updated_at = ? WHERE consolidation_id = ? AND state = 'committed'",
                        (
                            json.dumps(output, ensure_ascii=False, sort_keys=True),
                            datetime.now(UTC).isoformat(),
                            consolidation_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("Consolidation Manifest 状态不再是 committed")
                    connection.execute("COMMIT")
                except BaseException:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
                finally:
                    connection.close()
                entries.extend(implicit_entries)
            for index, raw in enumerate(entries):
                summary = str(raw.get("summary") or "").strip()
                kind = str(raw.get("kind") or "event").strip()
                if not summary or kind not in {"event", "profile", "preference", "procedure"}:
                    continue
                source_ref = f"consolidation:{consolidation_id}#{index}"
                with connect_database(self.store.database) as connection:
                    source = connection.execute(
                        "SELECT item_id FROM memory_sources WHERE source_ref = ?", (source_ref,)
                    ).fetchone()
                if source is not None:
                    counts["unchanged"] += 1
                    continue
                result = await self.memorizer.remember(
                    summary=summary,
                    memory_kind=kind,
                    source_ref=source_ref,
                    scope_channel=scope_channel,
                    scope_chat_id=scope_chat_id,
                    extra=object_dict(raw.get("extra")),
                    happened_at=optional_text(raw.get("happened_at")),
                    emotional_weight=bounded_int(raw.get("emotional_weight")),
                    explicit_supersedes=optional_text(raw.get("supersedes")),
                    assert_current=guard,
                )
                counts[result.status] += 1
            now = datetime.now(UTC).isoformat()
            guard()
            with nullcontext():
                with connect_database(self.store.database) as connection:
                    connection.execute(
                        "UPDATE memory_ingestion_batches SET state = 'committed', "
                        "committed_at = ?, updated_at = ? WHERE batch_id = ?",
                        (now, now, batch_id),
                    )
        except Exception as exc:
            with nullcontext():
                with connect_database(self.store.database) as connection:
                    connection.execute(
                        "UPDATE memory_ingestion_batches SET state = 'failed', last_error = ?, "
                        "updated_at = ? WHERE batch_id = ? AND state IN ('pending', 'writing')",
                        (str(exc), datetime.now(UTC).isoformat(), batch_id),
                    )
            raise
        return VectorizationResult(**counts)


def _event_entries(value: object, *, display_timezone: ZoneInfo) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, object]] = []
    for raw in value:
        if isinstance(raw, str):
            summary = raw.strip()
            weight = 0
        elif isinstance(raw, dict):
            summary = str(raw.get("summary") or "").strip()
            weight = bounded_int(raw.get("emotional_weight"))
        else:
            continue
        if not summary:
            continue
        happened_at: str | None = None
        match = re.match(r"^\[(\d{4}-\d{2}-\d{2})(?:\s+(\d{2}:\d{2}))?]", summary)
        if match is not None:
            local = datetime.fromisoformat(
                f"{match.group(1)}T{match.group(2) or '00:00'}:00"
            ).replace(tzinfo=display_timezone)
            happened_at = local.astimezone(UTC).isoformat()
        result.append(
            {
                "kind": "event",
                "summary": summary,
                "emotional_weight": weight,
                "happened_at": happened_at,
                "extra": {},
            }
        )
    return result


__all__ = ["ImplicitMemoryExtractor", "VectorizationResult", "VectorizationService"]
