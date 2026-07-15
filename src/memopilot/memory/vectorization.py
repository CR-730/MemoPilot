"""将已提交的 Consolidation 隐式记忆幂等写入 memory2.db。"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from memopilot.memory.contracts import EmbeddingProvider
from memopilot.memory.store import MemoryStore
from memopilot.persistence.migrations import connect_database
from memopilot.tasks.operational import LostLeaseError


@dataclass(frozen=True, slots=True)
class VectorizationResult:
    created: int = 0
    reinforced: int = 0
    unchanged: int = 0


class VectorizationService:
    def __init__(
        self,
        operational_database: Path,
        store: MemoryStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self.operational_database = operational_database
        self.store = store
        self.embedder = embedder

    async def run(
        self,
        consolidation_id: str,
        *,
        assert_current: Callable[[], None] | None = None,
        fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> VectorizationResult:
        guard = assert_current or (lambda: None)
        write_scope = fenced_write or nullcontext
        guard()
        with connect_database(self.operational_database) as connection:
            manifest = connection.execute(
                "SELECT model_output_json, state FROM consolidation_manifests "
                "WHERE consolidation_id = ?",
                (consolidation_id,),
            ).fetchone()
        if manifest is None:
            raise KeyError(consolidation_id)
        if str(manifest["state"]) != "committed":
            raise RuntimeError("只有已提交的 Consolidation 才能向量化")
        output = json.loads(str(manifest["model_output_json"]))
        memories = output.get("memories", []) if isinstance(output, dict) else []
        entries = [entry for entry in memories if isinstance(entry, dict)]
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
        counts = {"created": 0, "reinforced": 0, "unchanged": 0}
        try:
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
                embedding = await self.embedder.embed(summary)
                guard()
                with write_scope():
                    write = self.store.write_item(
                        summary=summary,
                        memory_type=kind,
                        source_ref=source_ref,
                        embedding=embedding,
                        happened_at=_optional_text(raw.get("happened_at")),
                        emotional_weight=_bounded_int(raw.get("emotional_weight")),
                        extra=_object(raw.get("extra")),
                    )
                    counts[write.status] += 1
                    supersedes = _optional_text(raw.get("supersedes"))
                    if supersedes:
                        self.store.mark_superseded(supersedes)
            now = datetime.now(UTC).isoformat()
            guard()
            with write_scope():
                with connect_database(self.store.database) as connection:
                    connection.execute(
                        "UPDATE memory_ingestion_batches SET state = 'committed', "
                        "committed_at = ?, updated_at = ? WHERE batch_id = ?",
                        (now, now, batch_id),
                    )
        except LostLeaseError:
            raise
        except Exception as exc:
            with write_scope():
                with connect_database(self.store.database) as connection:
                    connection.execute(
                        "UPDATE memory_ingestion_batches SET state = 'failed', last_error = ?, "
                        "updated_at = ? WHERE batch_id = ? AND state IN ('pending', 'writing')",
                        (str(exc), datetime.now(UTC).isoformat(), batch_id),
                    )
            raise
        return VectorizationResult(**counts)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _bounded_int(value: object) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        number = 0
    return max(0, min(10, number))


def _object(value: object) -> dict[str, object]:
    return {str(key): item for key, item in value.items()} if isinstance(value, dict) else {}


__all__ = ["VectorizationResult", "VectorizationService"]
