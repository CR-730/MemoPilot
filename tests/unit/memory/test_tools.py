from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime

from memopilot.memory.contracts import MemoryQueryResult, MemoryRecord
from memopilot.memory.store import MemoryStore
from memopilot.memory.tool_context import bind_memory_tool_context, reset_memory_tool_context
from memopilot.memory.tools import (
    build_forget_memory_tool,
    build_memorize_tool,
    build_recall_memory_tool,
)
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import ToolRegistry


class _Engine:
    def __init__(self) -> None:
        self.requests = []

    async def query(self, request):
        self.requests.append(request)
        return MemoryQueryResult(
            records=(MemoryRecord("m1", "event", "阶段四完成", 0.8),),
            trace={"intent": request.intent},
        )


async def test_recall_memory_tool_exposes_intents_and_structured_evidence() -> None:
    engine = _Engine()
    registry = ToolRegistry([build_recall_memory_tool(engine)])  # type: ignore[arg-type]

    observation = await registry.execute(
        FunctionCall(
            id="call-1",
            name="recall_memory",
            arguments={"query": "做完了什么", "intent": "answer", "limit": 4},
        )
    )
    assert observation.ok is True
    assert engine.requests[0].text == "做完了什么"
    assert engine.requests[0].intent == "answer"
    assert observation.result["records"][0]["id"] == "m1"
    assert observation.result["citation_required"] is True
    assert observation.result["cited_item_ids"] == ["m1"]


async def test_recall_memory_tool_accepts_friendly_time_filter() -> None:
    engine = _Engine()
    registry = ToolRegistry([build_recall_memory_tool(engine)])  # type: ignore[arg-type]

    observation = await registry.execute(
        FunctionCall(
            id="call-friendly-time",
            name="recall_memory",
            arguments={"query": "最近发生了什么", "time_filter": "recent_3d"},
        )
    )
    assert observation.ok is True
    assert engine.requests[0].time_start is not None
    assert engine.requests[0].time_end is not None
    assert 2.9 < (
        engine.requests[0].time_end - engine.requests[0].time_start
    ).total_seconds() / 86400 < 3.1


async def test_recall_memory_tool_accepts_multiple_memory_kinds() -> None:
    engine = _Engine()
    registry = ToolRegistry([build_recall_memory_tool(engine)])  # type: ignore[arg-type]

    observation = await registry.execute(
        FunctionCall(
            id="call-kinds",
            name="recall_memory",
            arguments={
                "query": "我的偏好和经历",
                "memory_kinds": ["preference", "event"],
            },
        )
    )

    assert observation.ok is True
    assert engine.requests[0].memory_kinds == ("preference", "event")


class _Memorizer:
    def __init__(self) -> None:
        self.calls = []

    async def remember(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "Result",
            (),
            {"item_id": "mem-1", "status": "created", "actual_kind": "procedure"},
        )()


async def test_memorize_tool_routes_explicit_memory_through_shared_writer() -> None:
    memorizer = _Memorizer()
    registry = ToolRegistry([build_memorize_tool(memorizer)])  # type: ignore[arg-type]

    token = bind_memory_tool_context(
        "feishu:chat-1",
        assert_current=lambda: None,
        fenced_write=nullcontext,
    )
    try:
        observation = await registry.execute(
            FunctionCall(
                id="call-memorize",
                name="memorize",
                arguments={
                    "summary": "发送邮件前先让我确认",
                    "memory_kind": "procedure",
                    "tool_requirement": "send_email",
                    "steps": ["展示草稿", "等待确认", "发送"],
                },
            )
        )
    finally:
        reset_memory_tool_context(token)

    assert observation.ok is True
    assert observation.result == {
        "item_id": "mem-1",
        "status": "created",
        "actual_kind": "procedure",
    }
    assert memorizer.calls[0]["scope_channel"] == "feishu"
    assert memorizer.calls[0]["extra"]["tool_requirement"] == "send_email"
    assert memorizer.calls[0]["source_ref"].startswith("explicit:")
    assert callable(memorizer.calls[0]["assert_current"])
    assert memorizer.calls[0]["fenced_write"] is nullcontext


async def test_recall_memory_tool_parses_timeline_boundaries() -> None:
    engine = _Engine()
    registry = ToolRegistry([build_recall_memory_tool(engine)])  # type: ignore[arg-type]

    observation = await registry.execute(
        FunctionCall(
            id="call-2",
            name="recall_memory",
            arguments={
                "query": "阶段记录",
                "intent": "timeline",
                "time_start": "2026-07-01T00:00:00+00:00",
                "time_end": "2026-08-01T00:00:00+00:00",
            },
        )
    )

    assert observation.ok is True
    assert engine.requests[0].time_start == datetime(2026, 7, 1, tzinfo=UTC)
    assert engine.requests[0].time_end == datetime(2026, 8, 1, tzinfo=UTC)


async def test_forget_memory_marks_known_items_superseded_and_reports_missing(tmp_path) -> None:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    store = MemoryStore(database, dimension=2, vector_enabled=False)
    created = store.write_item(
        summary="已经失效的偏好",
        memory_type="preference",
        source_ref="turn:forget",
        embedding=[1.0, 0.0],
    )
    registry = ToolRegistry([build_forget_memory_tool(store)])

    observation = await registry.execute(
        FunctionCall(
            id="call-forget",
            name="forget_memory",
            arguments={"ids": [created.item_id, "missing", created.item_id]},
        )
    )

    assert observation.ok is True
    assert observation.result == {
        "superseded_ids": [created.item_id],
        "missing_ids": ["missing"],
    }
    assert store.get_item(created.item_id)["status"] == "superseded"  # type: ignore[index]


async def test_forget_memory_does_not_write_after_lost_fence(tmp_path) -> None:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    store = MemoryStore(database, dimension=2, vector_enabled=False)
    created = store.write_item(
        summary="仍然有效的偏好",
        memory_type="preference",
        source_ref="turn:keep",
        embedding=[1.0, 0.0],
    )
    registry = ToolRegistry([build_forget_memory_tool(store)])

    def lost_fence() -> None:
        raise RuntimeError("lost lease")

    token = bind_memory_tool_context("feishu:chat-1", assert_current=lost_fence)
    try:
        observation = await registry.execute(
            FunctionCall(
                id="call-forget-lost",
                name="forget_memory",
                arguments={"ids": [created.item_id]},
            )
        )
    finally:
        reset_memory_tool_context(token)

    assert observation.ok is False
    assert store.get_item(created.item_id)["status"] == "active"  # type: ignore[index]


async def test_forget_memory_cannot_supersede_another_chat_but_can_forget_global(
    tmp_path,
) -> None:
    database = tmp_path / "memory2.db"
    migrate_database(database, DatabaseKind.MEMORY)
    store = MemoryStore(database, dimension=2, vector_enabled=False)
    other = store.write_item(
        summary="其他会话偏好",
        memory_type="preference",
        source_ref="other",
        embedding=[1.0, 0.0],
        scope_channel="feishu",
        scope_chat_id="chat-2",
    )
    global_item = store.write_item(
        summary="全局偏好",
        memory_type="preference",
        source_ref="global",
        embedding=[1.0, 0.0],
    )
    registry = ToolRegistry([build_forget_memory_tool(store)])

    token = bind_memory_tool_context("feishu:chat-1")
    try:
        observation = await registry.execute(
            FunctionCall(
                id="call-forget-scope",
                name="forget_memory",
                arguments={"ids": [other.item_id, global_item.item_id]},
            )
        )
    finally:
        reset_memory_tool_context(token)

    assert observation.ok is True
    assert observation.result["superseded_ids"] == [global_item.item_id]
    assert store.get_item(other.item_id)["status"] == "active"  # type: ignore[index]
    assert store.get_item(global_item.item_id)["status"] == "superseded"  # type: ignore[index]
