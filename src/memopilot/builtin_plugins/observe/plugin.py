from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from memopilot.bus.events import TurnCommitted
from memopilot.extensions.events import EventSubscription
from memopilot.extensions.plugin_base import Plugin
from memopilot.memory.events import MemoryWritten, RetrievalCompleted

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_key TEXT NOT NULL,
    user_msg TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    tool_chain TEXT,
    react_cache_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    react_cache_hit_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_turns_session_ts ON turns(session_key, ts);
CREATE TABLE IF NOT EXISTS rag_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_key TEXT NOT NULL,
    query TEXT NOT NULL,
    intent TEXT NOT NULL,
    aux_queries TEXT,
    hits_json TEXT,
    injected_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_rag_session_ts ON rag_queries(session_key, ts);
CREATE TABLE IF NOT EXISTS memory_writes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_key TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    item_id TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_memory_writes_session_ts
ON memory_writes(session_key, ts);
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.executescript(_SCHEMA)
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(turns)").fetchall()
    }
    for name in ("react_cache_prompt_tokens", "react_cache_hit_tokens"):
        if name not in columns:
            connection.execute(
                f"ALTER TABLE turns ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
            )
    connection.commit()
    return connection


class ObserveWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.queue: asyncio.Queue[object | None] = asyncio.Queue(maxsize=500)
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        connection = open_db(self.path)
        self.task = asyncio.create_task(
            self.run(connection),
            name="observe-writer",
        )

    def emit(self, event: object) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("observe queue full; event dropped")

    async def run(self, connection: sqlite3.Connection) -> None:
        try:
            while True:
                event = await self.queue.get()
                try:
                    if event is None:
                        return
                    try:
                        self._write(connection, event)
                    except Exception:
                        logger.exception("observe write failed")
                finally:
                    self.queue.task_done()
        finally:
            connection.close()

    async def close(self) -> None:
        if self.task is None:
            return
        await self.queue.join()
        await self.queue.put(None)
        await self.task
        self.task = None

    def _write(self, connection: sqlite3.Connection, event: object) -> None:
        ts = (
            _utc_iso(event.timestamp)
            if isinstance(event, TurnCommitted)
            else datetime.now(UTC).isoformat()
        )
        with connection:
            if isinstance(event, TurnCommitted):
                connection.execute(
                    """
                    INSERT INTO turns
                    (ts, session_key, user_msg, assistant_response, tool_chain,
                     react_cache_prompt_tokens, react_cache_hit_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        event.session_key,
                        event.input_message,
                        event.assistant_response,
                        json.dumps(event.tool_chain, ensure_ascii=False)
                        if event.tool_chain
                        else None,
                        event.react_cache_prompt_tokens,
                        event.react_cache_hit_tokens,
                    ),
                )
            elif isinstance(event, RetrievalCompleted):
                connection.execute(
                    """
                    INSERT INTO rag_queries
                    (ts, session_key, query, intent, aux_queries, hits_json,
                     injected_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        event.session_key,
                        event.query,
                        event.intent,
                        json.dumps(event.aux_queries, ensure_ascii=False)
                        if event.aux_queries
                        else None,
                        json.dumps(
                            [
                                {
                                    "id": record.id,
                                    "kind": record.kind,
                                    "score": record.score,
                                    "summary": record.summary,
                                    "injected": record.injected,
                                }
                                for record in event.records
                            ],
                            ensure_ascii=False,
                        )
                        if event.records
                        else None,
                        event.injected_count,
                    ),
                )
            elif isinstance(event, MemoryWritten):
                connection.execute(
                    """
                    INSERT INTO memory_writes
                    (ts, session_key, source_ref, memory_type, item_id, status,
                     summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        event.session_key,
                        event.source_ref,
                        event.memory_type,
                        event.item_id,
                        event.status,
                        event.summary,
                    ),
                )


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _run_retention(path: Path) -> None:
    stamp = path.parent / ".last_cleanup"
    if stamp.exists() and time.time() - stamp.stat().st_mtime < 86400:
        return
    connection = open_db(path)
    try:
        with connection:
            connection.execute(
                "DELETE FROM turns WHERE ts < datetime('now', '-180 days')"
            )
            connection.execute(
                "DELETE FROM rag_queries WHERE ts < datetime('now', '-90 days')"
            )
        stamp.write_text("ok", encoding="utf-8")
    finally:
        connection.close()


class ObservePlugin(Plugin):
    name = "observe"

    async def initialize(self) -> None:
        if self.context.workspace is None:
            return
        self.writer = ObserveWriter(
            self.context.workspace / "observe" / "observe.db"
        )
        self.writer.start()
        self.subscriptions: tuple[EventSubscription[object], ...] = (
            self.context.event_bus.on(
                TurnCommitted,
                self.writer.emit,
                observer=True,
                handler_id="observe.turn",
            ),
            self.context.event_bus.on(
                RetrievalCompleted,
                self.writer.emit,
                observer=True,
                handler_id="observe.retrieval",
            ),
            self.context.event_bus.on(
                MemoryWritten,
                self.writer.emit,
                observer=True,
                handler_id="observe.memory",
            ),
        )
        self.retention_task = asyncio.create_task(
            asyncio.to_thread(_run_retention, self.writer.path),
            name="observe-retention",
        )

    async def terminate(self) -> None:
        for subscription in getattr(self, "subscriptions", ()):
            subscription.unsubscribe()
        writer = getattr(self, "writer", None)
        if writer is not None:
            await writer.close()
        retention_task = getattr(self, "retention_task", None)
        if retention_task is not None:
            await retention_task
