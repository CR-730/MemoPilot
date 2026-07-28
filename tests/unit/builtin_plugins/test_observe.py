from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot.bus.events import TurnCommitted
from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.memory.contracts import MemoryQuery, MemoryRecord
from memopilot.memory.engine import LayeredMemoryEngine
from memopilot.memory.events import MemoryWritten, RetrievalCompleted
from memopilot.memory.memorizer import MemoryMemorizer
from memopilot.memory.store import MemoryStore
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.tools import ToolRegistry

_PLUGIN_DIR = (
    Path(__file__).parents[3] / "src" / "memopilot" / "builtin_plugins" / "observe"
)


async def _manager(tmp_path: Path, bus: EventBus) -> PluginManager:
    manager = PluginManager(
        [_PLUGIN_DIR],
        event_bus=bus,
        tool_registry=ToolRegistry(),
        workspace=tmp_path,
    )
    await manager.load_all()
    assert manager.loaded_plugin_ids == ("observe",)
    return manager


async def test_observe_persists_three_event_types_and_drains_on_shutdown(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    manager = await _manager(tmp_path, bus)
    now = datetime.now(UTC)

    await bus.fanout(
        TurnCommitted(
            "feishu:chat",
            "feishu",
            "chat",
            "你好",
            "收到",
            ["shell"],
            now,
            ({"name": "shell", "status": "success", "result": "ok"},),
        )
    )
    await bus.fanout(
        RetrievalCompleted(
            "feishu:chat",
            "问题",
            "answer",
            (MemoryRecord("m1", "event", "摘要", 0.9, injected=True),),
            ("假设",),
            1,
        )
    )
    await bus.fanout(
        MemoryWritten(
            "feishu:chat",
            "turn:1",
            "event",
            "m2",
            "created",
            "新记忆",
        )
    )

    await manager.unload_all()
    connection = sqlite3.connect(tmp_path / "observe" / "observe.db")
    try:
        assert connection.execute("SELECT count(*) FROM turns").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM rag_queries").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM memory_writes").fetchone() == (1,)
        assert connection.execute(
            "SELECT ts, tool_chain FROM turns"
        ).fetchone() == (
            now.isoformat(),
            '[{"name": "shell", "status": "success", "result": "ok"}]',
        )
    finally:
        connection.close()
        await bus.aclose()


async def test_observe_write_failure_drops_one_event_then_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()
    manager = await _manager(tmp_path, bus)
    plugin = manager.get_plugin("observe")
    writer = plugin.writer
    real_write = writer._write
    calls = 0

    def flaky_write(connection: sqlite3.Connection, event: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("fail once")
        real_write(connection, event)

    monkeypatch.setattr(writer, "_write", flaky_write)
    for content in ("one", "two"):
        await bus.fanout(
            TurnCommitted(
                "cli:chat",
                "cli",
                "chat",
                content,
                content,
                [],
                datetime.now(UTC),
            )
        )

    await manager.unload_all()
    connection = sqlite3.connect(tmp_path / "observe" / "observe.db")
    try:
        assert connection.execute(
            "SELECT assistant_response FROM turns"
        ).fetchall() == [("two",)]
    finally:
        connection.close()
        await bus.aclose()


async def test_observe_open_failure_is_not_loaded_and_shutdown_does_not_hang(
    tmp_path: Path,
) -> None:
    (tmp_path / "observe").write_text("占用目录路径", encoding="utf-8")
    bus = EventBus()
    manager = PluginManager(
        [_PLUGIN_DIR],
        event_bus=bus,
        tool_registry=ToolRegistry(),
        workspace=tmp_path,
    )

    await manager.load_all()

    assert manager.loaded_plugin_ids == ()
    assert manager.diagnostics[-1].code == "initialization_failed"
    await manager.unload_all()
    await bus.aclose()


async def test_timeline_query_emits_once_and_failure_emits_nothing() -> None:
    class Store:
        fail = False

        def list_events(self, **_: object) -> list[dict[str, object]]:
            if self.fail:
                raise RuntimeError("boom")
            return []

    store = Store()
    bus = EventBus()
    events: list[RetrievalCompleted] = []
    bus.on(RetrievalCompleted, events.append, observer=True)
    engine = LayeredMemoryEngine(
        SimpleNamespace(store=store),
        event_bus=bus,
    )
    now = datetime.now(UTC)
    request = MemoryQuery(
        "最近",
        intent="timeline",
        session_key="cli:chat",
        time_start=now - timedelta(days=1),
        time_end=now,
    )

    await engine.query(request)
    assert len(events) == 1
    store.fail = True
    with pytest.raises(RuntimeError, match="boom"):
        await engine.query(request)
    assert len(events) == 1
    await bus.aclose()


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


async def test_remember_success_paths_each_emit_once(tmp_path: Path) -> None:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    bus = EventBus()
    events: list[MemoryWritten] = []
    bus.on(MemoryWritten, events.append, observer=True)
    memorizer = MemoryMemorizer(
        MemoryStore(database, dimension=2, vector_enabled=False),
        _Embedder(),
        event_bus=bus,
    )

    await memorizer.remember(
        summary="用户完成阶段",
        memory_kind="event",
        source_ref="turn:1",
        scope_channel="cli",
        scope_chat_id="chat",
    )
    await memorizer.remember(
        summary="用户完成阶段",
        memory_kind="event",
        source_ref="turn:2",
        scope_channel="cli",
        scope_chat_id="chat",
    )
    await memorizer.remember(
        summary="发送邮件前先展示草稿",
        memory_kind="procedure",
        source_ref="turn:3",
        scope_channel="cli",
        scope_chat_id="chat",
        extra={"tool_requirement": "send_email", "steps": ["展示草稿"]},
    )
    await memorizer.remember(
        summary="发送邮件前等待确认",
        memory_kind="procedure",
        source_ref="turn:4",
        scope_channel="cli",
        scope_chat_id="chat",
        extra={"tool_requirement": "send_email", "steps": ["等待确认"]},
    )

    assert [event.status for event in events] == [
        "created",
        "reinforced",
        "created",
        "merged",
    ]
    await bus.aclose()


async def test_retention_skips_memory_writes_and_runs_at_most_daily(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    manager = await _manager(tmp_path, bus)
    plugin = manager.get_plugin("observe")
    await manager.unload_all()
    module = sys.modules[type(plugin).__module__]
    path = tmp_path / "observe" / "observe.db"
    (path.parent / ".last_cleanup").unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    old = "2000-01-01T00:00:00+00:00"
    try:
        connection.execute(
            "INSERT INTO turns(ts, session_key, user_msg, assistant_response) "
            "VALUES (?, 's', 'u', 'a')",
            (old,),
        )
        connection.execute(
            "INSERT INTO rag_queries"
            "(ts, session_key, query, intent, injected_count) "
            "VALUES (?, 's', 'q', 'answer', 0)",
            (old,),
        )
        connection.execute(
            "INSERT INTO memory_writes"
            "(ts, session_key, source_ref, memory_type, item_id, status, summary) "
            "VALUES (?, 's', 'r', 'event', 'm', 'created', 'x')",
            (old,),
        )
        connection.commit()
    finally:
        connection.close()

    module._run_retention(path)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count(*) FROM turns").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM rag_queries").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM memory_writes").fetchone() == (1,)
        connection.execute(
            "INSERT INTO turns(ts, session_key, user_msg, assistant_response) "
            "VALUES (?, 's', 'u', 'a')",
            (old,),
        )
        connection.commit()
    finally:
        connection.close()
    module._run_retention(path)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count(*) FROM turns").fetchone() == (1,)
    finally:
        connection.close()
        await bus.aclose()
