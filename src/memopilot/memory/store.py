"""memory2.db 的事实仓储与可重建检索索引。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from memopilot.memory.procedures import build_procedure_rule_schema, build_trigger_tags
from memopilot.persistence.migrations import connect_database


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    item_id: str
    status: str


class MemoryStore:
    def __init__(
        self,
        database: Path,
        *,
        dimension: int,
        vector_enabled: bool = True,
        hotness_alpha: float = 0.2,
        hotness_half_life_days: float = 14.0,
    ) -> None:
        if dimension < 1:
            raise ValueError("向量维度必须大于 0")
        self.database = Path(database)
        self.dimension = dimension
        self.hotness_alpha = max(0.0, min(1.0, hotness_alpha))
        self.hotness_half_life_days = max(1.0, hotness_half_life_days)
        self._vector_requested = vector_enabled
        self._vector_available = False
        self._initialize_indexes()

    @property
    def vector_available(self) -> bool:
        return self._vector_available

    def write_item(
        self,
        *,
        summary: str,
        memory_type: str,
        source_ref: str,
        embedding: list[float],
        happened_at: str | None = None,
        emotional_weight: int = 0,
        extra: dict[str, object] | None = None,
        supersede_ids: tuple[str, ...] = (),
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> MemoryWriteResult:
        normalized = _normalize_text(summary)
        if not normalized or not source_ref.strip():
            raise ValueError("记忆摘要和 source_ref 不能为空")
        self._validate_embedding(embedding)
        now = datetime.now(UTC).isoformat()
        content_hash = _content_hash(
            normalized,
            memory_type,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            happened_at=happened_at,
        )
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing_source = connection.execute(
                "SELECT item_id FROM memory_sources WHERE source_ref = ?",
                (source_ref,),
            ).fetchone()
            if existing_source is not None:
                connection.execute("COMMIT")
                return MemoryWriteResult(str(existing_source["item_id"]), "unchanged")

            exact = connection.execute(
                """
                SELECT item_id FROM memory_items
                WHERE memory_type = ? AND content_hash = ?
                  AND COALESCE(json_extract(extra_json, '$.scope_channel'), '') = ?
                  AND COALESCE(json_extract(extra_json, '$.scope_chat_id'), '') = ?
                """,
                (memory_type, content_hash, scope_channel, scope_chat_id),
            ).fetchone()
            if exact is not None:
                item_id = str(exact["item_id"])
                connection.execute(
                    """
                    UPDATE memory_items
                    SET status = 'active', reinforcement_count = reinforcement_count + 1,
                        emotional_weight = MAX(emotional_weight, ?), updated_at = ?
                    WHERE item_id = ?
                    """,
                    (max(0, min(10, emotional_weight)), now, item_id),
                )
                connection.execute(
                    "INSERT INTO memory_sources(source_ref, item_id, created_at) VALUES (?, ?, ?)",
                    (source_ref, item_id, now),
                )
                connection.execute("COMMIT")
                return MemoryWriteResult(item_id, "reinforced")

            item_id = str(uuid5(NAMESPACE_URL, f"memopilot:memory:{source_ref}"))
            stored_extra = dict(extra or {})
            if scope_channel:
                stored_extra["scope_channel"] = scope_channel
            if scope_chat_id:
                stored_extra["scope_chat_id"] = scope_chat_id
            connection.execute(
                """
                INSERT INTO memory_items(
                    item_id, memory_type, summary, content_hash, source_ref, status,
                    reinforcement_count, emotional_weight, happened_at, extra_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'active', 1, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    memory_type,
                    normalized,
                    content_hash,
                    source_ref,
                    max(0, min(10, emotional_weight)),
                    happened_at,
                    json.dumps(stored_extra, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            blob = _pack_vector(embedding)
            connection.execute(
                """
                INSERT INTO memory_embeddings(
                    item_id, provider, model, dimension, embedding, content_hash,
                    created_at, updated_at
                ) VALUES (?, 'configured', 'configured', ?, ?, ?, ?, ?)
                """,
                (item_id, self.dimension, blob, content_hash, now, now),
            )
            connection.execute(
                "INSERT INTO memory_sources(source_ref, item_id, created_at) VALUES (?, ?, ?)",
                (source_ref, item_id, now),
            )
            retired = tuple(dict.fromkeys(value for value in supersede_ids if value != item_id))
            if retired:
                placeholders = ",".join("?" for _ in retired)
                connection.execute(
                    f"UPDATE memory_items SET status = 'superseded', updated_at = ? "
                    f"WHERE item_id IN ({placeholders}) AND status = 'active' "
                    "AND memory_type = ? AND ("
                    "(? = '' AND ? = '') "
                    "OR (COALESCE(json_extract(extra_json, '$.scope_channel'), '') = '' "
                    "    AND COALESCE(json_extract(extra_json, '$.scope_chat_id'), '') = '') "
                    "OR ((? = '' OR json_extract(extra_json, '$.scope_channel') = ?) "
                    "    AND (? = '' OR json_extract(extra_json, '$.scope_chat_id') = ?))"
                    ")",
                    (
                        now,
                        *retired,
                        memory_type,
                        scope_channel,
                        scope_chat_id,
                        scope_channel,
                        scope_channel,
                        scope_chat_id,
                        scope_chat_id,
                    ),
                )
            row = connection.execute(
                "INSERT INTO memory_vector_rows(item_id) VALUES (?) RETURNING row_id",
                (item_id,),
            ).fetchone()
            assert row is not None
            row_id = int(row["row_id"])
            if self._vector_available:
                connection.execute(
                    "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
                    (row_id, json.dumps(embedding, separators=(",", ":"))),
                )
            connection.execute(
                "INSERT INTO memory_fts(item_id, terms) VALUES (?, ?)",
                (item_id, " ".join(_extract_terms(normalized))),
            )
            connection.execute("COMMIT")
            return MemoryWriteResult(item_id, "created")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_item(self, item_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_items WHERE item_id = ?", (item_id,)
            ).fetchone()
        return None if row is None else _item_dict(row)

    def count_sources(self, item_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM memory_sources WHERE item_id = ?", (item_id,)
            ).fetchone()
        assert row is not None
        return int(row[0])

    def reinforce_with_source(
        self,
        item_id: str,
        *,
        source_ref: str,
        emotional_weight: int = 0,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            exists = connection.execute(
                "SELECT 1 FROM memory_sources WHERE source_ref = ?", (source_ref,)
            ).fetchone()
            if exists is not None:
                connection.execute("COMMIT")
                return False
            connection.execute(
                "INSERT INTO memory_sources(source_ref, item_id, created_at) VALUES (?, ?, ?)",
                (source_ref, item_id, now),
            )
            connection.execute(
                "UPDATE memory_items SET status = 'active', "
                "reinforcement_count = reinforcement_count + 1, "
                "emotional_weight = MAX(emotional_weight, ?), updated_at = ? "
                "WHERE item_id = ?",
                (max(0, min(10, emotional_weight)), now, item_id),
            )
            connection.execute("COMMIT")
            return True
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def reinforce_items_once(self, item_ids: tuple[str, ...], *, usage_ref: str) -> int:
        """按一次可审计使用记录强化记忆；同一 usage_ref 重放不会重复计数。"""
        clean_ids = tuple(dict.fromkeys(item_id.strip() for item_id in item_ids if item_id.strip()))
        if not clean_ids or not usage_ref.strip():
            return 0
        now = datetime.now(UTC).isoformat()
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        changed = 0
        try:
            for item_id in clean_ids:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO memory_usages(usage_ref, item_id, created_at) "
                    "SELECT ?, item_id, ? FROM memory_items "
                    "WHERE item_id = ? AND status = 'active'",
                    (usage_ref, now, item_id),
                )
                if cursor.rowcount != 1:
                    continue
                connection.execute(
                    "UPDATE memory_items SET reinforcement_count = reinforcement_count + 1, "
                    "updated_at = ? WHERE item_id = ?",
                    (now, item_id),
                )
                changed += 1
            connection.execute("COMMIT")
            return changed
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def merge_item(
        self,
        item_id: str,
        *,
        summary: str,
        source_ref: str,
        embedding: list[float],
        extra: dict[str, object],
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> MemoryWriteResult:
        normalized = _normalize_text(summary)
        self._validate_embedding(embedding)
        now = datetime.now(UTC).isoformat()
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing_source = connection.execute(
                "SELECT item_id FROM memory_sources WHERE source_ref = ?", (source_ref,)
            ).fetchone()
            if existing_source is not None:
                connection.execute("COMMIT")
                return MemoryWriteResult(str(existing_source["item_id"]), "unchanged")
            existing_item = connection.execute(
                "SELECT extra_json FROM memory_items "
                "WHERE item_id = ? AND memory_type = 'procedure'",
                (item_id,),
            ).fetchone()
            if existing_item is None:
                raise KeyError(item_id)
            old_extra = json.loads(str(existing_item["extra_json"]) or "{}")
            merged_extra = dict(old_extra) if isinstance(old_extra, dict) else {}
            merged_extra.update(extra)
            for field in ("scope_channel", "scope_chat_id"):
                if old_extra.get(field):
                    merged_extra[field] = old_extra[field]
            content_hash = _content_hash(
                normalized,
                "procedure",
                scope_channel=str(merged_extra.get("scope_channel") or ""),
                scope_chat_id=str(merged_extra.get("scope_chat_id") or ""),
            )
            old_steps_value = old_extra.get("steps")
            new_steps_value = extra.get("steps")
            old_steps = old_steps_value if isinstance(old_steps_value, list) else []
            new_steps = new_steps_value if isinstance(new_steps_value, list) else []
            steps = list(
                dict.fromkeys(
                    str(step).strip()
                    for step in (*old_steps, *new_steps)
                    if str(step).strip()
                )
            )
            requirement = str(
                extra.get("tool_requirement") or old_extra.get("tool_requirement") or ""
            ).strip()
            old_schema = old_extra.get("rule_schema")
            new_schema = extra.get("rule_schema")
            combined_values: dict[str, list[str]] = {
                "required_tools": [],
                "forbidden_tools": [],
                "mentioned_tools": [],
            }
            for schema in (old_schema, new_schema):
                if not isinstance(schema, dict):
                    continue
                for field in ("required_tools", "forbidden_tools", "mentioned_tools"):
                    values = schema.get(field)
                    if isinstance(values, list):
                        combined_values[field].extend(str(value) for value in values)
            combined_schema: dict[str, object] = dict(combined_values)
            rule_schema = build_procedure_rule_schema(
                normalized,
                tool_requirement=requirement or None,
                steps=steps,
                rule_schema=combined_schema,
            )
            schema_for_tags: dict[str, object] = dict(rule_schema)
            merged_extra.update(
                {
                    "steps": steps,
                    "rule_schema": rule_schema,
                    "trigger_tags": build_trigger_tags(
                        normalized,
                        tool_requirement=requirement or None,
                        rule_schema=schema_for_tags,
                    ),
                    "_merge_note": "合并同一执行条件的 Procedure 记忆",
                }
            )
            if requirement:
                merged_extra["tool_requirement"] = requirement
            changed = connection.execute(
                "UPDATE memory_items SET summary = ?, content_hash = ?, status = 'active', "
                "reinforcement_count = reinforcement_count + 1, "
                "emotional_weight = MAX(emotional_weight, ?), "
                "happened_at = COALESCE(?, happened_at), extra_json = ?, updated_at = ? "
                "WHERE item_id = ? AND memory_type = 'procedure'",
                (
                    normalized,
                    content_hash,
                    max(0, min(10, emotional_weight)),
                    happened_at,
                    json.dumps(merged_extra, ensure_ascii=False, sort_keys=True),
                    now,
                    item_id,
                ),
            )
            if changed.rowcount != 1:
                raise KeyError(item_id)
            blob = _pack_vector(embedding)
            connection.execute(
                "UPDATE memory_embeddings SET embedding = ?, content_hash = ?, updated_at = ? "
                "WHERE item_id = ?",
                (blob, content_hash, now, item_id),
            )
            connection.execute("DELETE FROM memory_fts WHERE item_id = ?", (item_id,))
            connection.execute(
                "INSERT INTO memory_fts(item_id, terms) VALUES (?, ?)",
                (item_id, " ".join(_extract_terms(normalized))),
            )
            if self._vector_available:
                row = connection.execute(
                    "SELECT row_id FROM memory_vector_rows WHERE item_id = ?", (item_id,)
                ).fetchone()
                assert row is not None
                connection.execute("DELETE FROM memory_vectors WHERE rowid = ?", (int(row[0]),))
                connection.execute(
                    "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
                    (int(row[0]), json.dumps(embedding, separators=(",", ":"))),
                )
            connection.execute(
                "INSERT INTO memory_sources(source_ref, item_id, created_at) VALUES (?, ?, ?)",
                (source_ref, item_id, now),
            )
            connection.execute("COMMIT")
            return MemoryWriteResult(item_id, "merged")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def search_vectors(
        self,
        vector: list[float],
        *,
        limit: int,
        memory_types: tuple[str, ...] = (),
        score_threshold: float = 0.0,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> list[dict[str, object]]:
        self._validate_embedding(vector)
        if self._vector_available:
            candidate_limit = max(limit * 4, limit)
            while True:
                candidates = self._search_vec0(vector, candidate_limit)
                filtered = [
                    item
                    for item in candidates
                    if _matches_filters(
                        item,
                        memory_types,
                        time_start,
                        time_end,
                        scope_channel,
                        scope_chat_id,
                    )
                    and float(str(item["semantic_score"])) >= score_threshold
                ]
                if len(filtered) >= limit or len(candidates) < candidate_limit:
                    break
                candidate_limit *= 2
        else:
            filtered = [
                item
                for item in self._search_cosine(vector)
                if _matches_filters(
                    item,
                    memory_types,
                    time_start,
                    time_end,
                    scope_channel,
                    scope_chat_id,
                )
                and float(str(item["semantic_score"])) >= score_threshold
            ]
        return sorted(filtered, key=lambda item: float(str(item["score"])), reverse=True)[:limit]

    def search_keywords(
        self,
        query: str,
        *,
        limit: int,
        memory_types: tuple[str, ...] = (),
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> list[dict[str, object]]:
        terms = _extract_terms(query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        candidate_limit = max(limit * 4, limit)
        while True:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT i.*, bm25(memory_fts) AS keyword_rank
                    FROM memory_fts
                    JOIN memory_items AS i ON i.item_id = memory_fts.item_id
                    WHERE memory_fts MATCH ? AND i.status = 'active'
                    ORDER BY bm25(memory_fts)
                    LIMIT ?
                    """,
                    (expression, candidate_limit),
                ).fetchall()
            filtered = [
                item
                for row in rows
                if _matches_filters(
                    (item := _item_dict(row)),
                    memory_types,
                    time_start,
                    time_end,
                    scope_channel,
                    scope_chat_id,
                )
            ]
            if len(filtered) >= limit or len(rows) < candidate_limit:
                break
            candidate_limit *= 2
        for rank, item in enumerate(filtered, start=1):
            item["keyword_rank"] = rank
            item["score"] = 1.0 / rank
        return filtered[:limit]

    def list_events(
        self,
        *,
        time_start: datetime,
        time_end: datetime,
        limit: int,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memory_items
                WHERE memory_type = 'event' AND status = 'active'
                  AND happened_at IS NOT NULL
                  AND julianday(happened_at) >= julianday(?)
                  AND julianday(happened_at) < julianday(?)
                  AND (
                    COALESCE(json_extract(extra_json, '$.scope_channel'), '') = ''
                    OR (? = '' AND ? = '')
                    OR (
                      (? = '' OR json_extract(extra_json, '$.scope_channel') = ?)
                      AND (? = '' OR json_extract(extra_json, '$.scope_chat_id') = ?)
                    )
                  )
                ORDER BY julianday(happened_at) DESC LIMIT ?
                """,
                (
                    time_start.isoformat(),
                    time_end.isoformat(),
                    scope_channel,
                    scope_chat_id,
                    scope_channel,
                    scope_channel,
                    scope_chat_id,
                    scope_chat_id,
                    limit,
                ),
            ).fetchall()
        items = [
            item
            for row in rows
            if _matches_filters(
                (item := _item_dict(row)),
                (),
                time_start,
                time_end,
                scope_channel,
                scope_chat_id,
            )
        ]
        for item in items:
            item["score"] = 1.0
        return items

    def reinforce(self, item_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_items SET reinforcement_count = reinforcement_count + 1, "
                "updated_at = ? WHERE item_id = ?",
                (datetime.now(UTC).isoformat(), item_id),
            )

    def mark_superseded(self, item_id: str) -> None:
        self.mark_superseded_batch((item_id,))

    def mark_superseded_batch(
        self,
        item_ids: tuple[str, ...],
        *,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> None:
        unique_ids = tuple(dict.fromkeys(value for value in item_ids if value))
        if not unique_ids:
            return
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE memory_items SET status = 'superseded', updated_at = ? "
                f"WHERE item_id IN ({placeholders}) AND status = 'active' AND ("
                "(? = '' AND ? = '') "
                "OR (COALESCE(json_extract(extra_json, '$.scope_channel'), '') = '' "
                "    AND COALESCE(json_extract(extra_json, '$.scope_chat_id'), '') = '') "
                "OR ((? = '' OR json_extract(extra_json, '$.scope_channel') = ?) "
                "    AND (? = '' OR json_extract(extra_json, '$.scope_chat_id') = ?))"
                ")",
                (
                    datetime.now(UTC).isoformat(),
                    *unique_ids,
                    scope_channel,
                    scope_chat_id,
                    scope_channel,
                    scope_channel,
                    scope_chat_id,
                    scope_chat_id,
                ),
            )

    def _initialize_indexes(self) -> None:
        connection = connect_database(self.database)
        try:
            self._rebuild_missing_keyword_rows(connection)
            if self._vector_requested:
                _load_sqlite_vec(connection)
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors "
                    f"USING vec0(embedding float[{self.dimension}] distance_metric=cosine)"
                )
                self._vector_available = True
                self._rebuild_missing_vector_rows(connection)
        except (ImportError, sqlite3.Error):
            self._vector_available = False
        finally:
            connection.close()

    @staticmethod
    def _rebuild_missing_keyword_rows(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT i.item_id, i.summary
            FROM memory_items AS i
            LEFT JOIN memory_fts AS f ON f.item_id = i.item_id
            WHERE f.item_id IS NULL
            """
        ).fetchall()
        connection.executemany(
            "INSERT INTO memory_fts(item_id, terms) VALUES (?, ?)",
            ((str(row["item_id"]), " ".join(_extract_terms(str(row["summary"])))) for row in rows),
        )

    def _rebuild_missing_vector_rows(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT m.row_id, e.embedding
            FROM memory_vector_rows AS m
            JOIN memory_embeddings AS e ON e.item_id = m.item_id
            LEFT JOIN memory_vectors AS v ON v.rowid = m.row_id
            WHERE v.rowid IS NULL AND e.dimension = ?
            """,
            (self.dimension,),
        ).fetchall()
        connection.executemany(
            "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
            (
                (
                    int(row["row_id"]),
                    json.dumps(
                        _unpack_vector(row["embedding"], self.dimension),
                        separators=(",", ":"),
                    ),
                )
                for row in rows
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = connect_database(self.database)
        if self._vector_available:
            _load_sqlite_vec(connection)
        return connection

    def _search_vec0(self, vector: list[float], limit: int) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.*, v.distance
                FROM (
                    SELECT rowid, distance FROM memory_vectors
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance
                ) AS v
                JOIN memory_vector_rows AS m ON m.row_id = v.rowid
                JOIN memory_items AS i ON i.item_id = m.item_id
                WHERE i.status = 'active'
                """,
                (json.dumps(vector, separators=(",", ":")), limit),
            ).fetchall()
        return [
            _scored_item(
                row,
                1.0 - float(row["distance"]),
                hotness_alpha=self.hotness_alpha,
                hotness_half_life_days=self.hotness_half_life_days,
            )
            for row in rows
        ]

    def _search_cosine(self, vector: list[float]) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.*, e.embedding
                FROM memory_items AS i
                JOIN memory_embeddings AS e ON e.item_id = i.item_id
                WHERE i.status = 'active' AND e.dimension = ?
                """,
                (self.dimension,),
            ).fetchall()
        return [
            _scored_item(
                row,
                _cosine(vector, _unpack_vector(row["embedding"], self.dimension)),
                hotness_alpha=self.hotness_alpha,
                hotness_half_life_days=self.hotness_half_life_days,
            )
            for row in rows
        ]

    def _validate_embedding(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"向量维度不匹配: expected={self.dimension}, actual={len(vector)}")


def _load_sqlite_vec(connection: sqlite3.Connection) -> None:
    import sqlite_vec  # type: ignore[import-untyped]

    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _content_hash(
    summary: str,
    memory_type: str,
    *,
    scope_channel: str = "",
    scope_chat_id: str = "",
    happened_at: str | None = None,
) -> str:
    scope = f"\0{scope_channel}\0{scope_chat_id}" if scope_channel or scope_chat_id else ""
    event_time = f"\0{happened_at}" if memory_type == "event" and happened_at else ""
    return hashlib.sha256(
        f"{memory_type}\0{summary.casefold()}{scope}{event_time}".encode()
    ).hexdigest()


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(value: object, dimension: int) -> list[float]:
    if not isinstance(value, bytes | bytearray | memoryview):
        raise TypeError("向量列必须是二进制数据")
    return list(struct.unpack(f"<{dimension}f", bytes(value)))


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1e-9
    right_norm = math.sqrt(sum(value * value for value in right)) or 1e-9
    return dot / left_norm / right_norm


def _item_dict(row: sqlite3.Row) -> dict[str, object]:
    item = {
        "item_id": str(row["item_id"]),
        "memory_type": str(row["memory_type"]),
        "summary": str(row["summary"]),
        "content_hash": str(row["content_hash"]),
        "source_ref": str(row["source_ref"]),
        "status": str(row["status"]),
        "reinforcement_count": int(row["reinforcement_count"]),
        "emotional_weight": int(row["emotional_weight"]),
        "happened_at": None if row["happened_at"] is None else str(row["happened_at"]),
        "extra_json": json.loads(str(row["extra_json"] or "{}")),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }
    return item


def _hotness_score(
    reinforcement: int,
    updated_at: datetime,
    *,
    now: datetime | None = None,
    half_life_days: float = 14.0,
    emotional_weight: int = 0,
) -> float:
    current = now or datetime.now(UTC)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    bounded_emotion = max(0, min(10, emotional_weight))
    effective_half_life = max(
        half_life_days * (1.0 + 0.5 * bounded_emotion / 10.0),
        0.1,
    )
    frequency = 1.0 / (1.0 + math.exp(-math.log1p(max(0, reinforcement))))
    age_days = max((current - updated_at).total_seconds() / 86400.0, 0.0)
    recency = math.exp(-math.log(2) / effective_half_life * age_days)
    return frequency * recency


def _scored_item(
    row: sqlite3.Row,
    semantic_score: float,
    *,
    hotness_alpha: float,
    hotness_half_life_days: float,
) -> dict[str, object]:
    item = _item_dict(row)
    updated = datetime.fromisoformat(str(item["updated_at"]))
    reinforcement = int(str(item["reinforcement_count"]))
    hotness = _hotness_score(
        reinforcement,
        updated,
        half_life_days=hotness_half_life_days,
        emotional_weight=int(str(item["emotional_weight"])),
    )
    item["semantic_score"] = semantic_score
    item["hotness"] = hotness
    item["score"] = (1.0 - hotness_alpha) * semantic_score + hotness_alpha * hotness
    return item


def _matches_filters(
    item: dict[str, object],
    memory_types: tuple[str, ...],
    time_start: datetime | None,
    time_end: datetime | None,
    scope_channel: str = "",
    scope_chat_id: str = "",
) -> bool:
    if memory_types and str(item["memory_type"]) not in memory_types:
        return False
    extra = item.get("extra_json")
    if isinstance(extra, dict):
        item_channel = str(extra.get("scope_channel") or "")
        item_chat_id = str(extra.get("scope_chat_id") or "")
        if scope_channel and item_channel and item_channel != scope_channel:
            return False
        if scope_chat_id and item_chat_id and item_chat_id != scope_chat_id:
            return False
    happened = item.get("happened_at")
    if time_start is not None or time_end is not None:
        if not happened:
            return False
        value = _utc_datetime(datetime.fromisoformat(str(happened)))
        if time_start is not None and value < _utc_datetime(time_start):
            return False
        if time_end is not None and value >= _utc_datetime(time_end):
            return False
    return True


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


_STOPWORDS = {"用户", "助手", "这个", "那个", "什么", "如何", "是否", "没有", "当前", "最近"}


def _extract_terms(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_.-]{2,}", text)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(chunk) <= 4 and chunk not in _STOPWORDS:
            terms.append(chunk)
        terms.extend(
            chunk[index : index + 2]
            for index in range(len(chunk) - 1)
            if chunk[index : index + 2] not in _STOPWORDS
        )
    return list(dict.fromkeys(terms))[:40]


__all__ = ["MemoryStore", "MemoryWriteResult"]
