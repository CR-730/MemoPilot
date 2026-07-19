"""记忆工具的显式 Function Calling 合同。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from memopilot.memory.contracts import MemoryQuery, MemoryQueryEngine
from memopilot.memory.store import MemoryStore
from memopilot.memory.tool_context import current_memory_tool_context
from memopilot.runtime.tools import Tool

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_RECENT_PRESETS = {"recent_3d": 3, "recent_7d": 7, "recent_30d": 30}


class ExplicitMemorizer(Protocol):
    async def remember(
        self,
        *,
        summary: str,
        memory_kind: str,
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        extra: dict[str, object],
        assert_current: Callable[[], None] | None = None,
        fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> Any: ...


def build_recall_memory_tool(engine: MemoryQueryEngine) -> Tool:
    async def recall_memory(
        *,
        query: str,
        intent: str = "answer",
        limit: int = 8,
        time_start: str | None = None,
        time_end: str | None = None,
        time_filter: str = "",
        memory_kind: str = "",
        memory_kinds: list[str] | None = None,
    ) -> dict[str, object]:
        context = current_memory_tool_context()
        session_key = context.session_key if context is not None else ""
        friendly_window = _parse_time_filter(time_filter)
        if time_filter and friendly_window is None:
            return {"records": [], "trace": {}, "error": "invalid_time_filter"}
        selected_kinds = tuple(
            dict.fromkeys(
                value.strip() for value in (memory_kinds or []) if value.strip()
            )
        )
        if not selected_kinds and memory_kind.strip():
            selected_kinds = (memory_kind.strip(),)
        result = await engine.query(
            MemoryQuery(
                text=query,
                intent=intent,  # type: ignore[arg-type]
                session_key=session_key,
                memory_kinds=selected_kinds,
                limit=limit,
                time_start=friendly_window[0] if friendly_window else _parse_time(time_start),
                time_end=friendly_window[1] if friendly_window else _parse_time(time_end),
            )
        )
        cited_ids = [record.id for record in result.records if record.id]
        return {
            "text_block": result.text_block,
            "records": [asdict(record) for record in result.records],
            "trace": result.trace,
            "citation_required": True,
            "citation_format": "§cited:[id1,id2,...]§",
            "cited_item_ids": cited_ids,
            "citation_rule": (
                "若最终回复使用了本工具返回的任何记忆条目，"
                "必须在正文末尾输出 §cited:[实际使用的id列表]§"
            ),
        }

    return Tool(
        name="recall_memory",
        description=(
            "检索长期记忆。回答问题时用 answer；按时间回顾用 timeline；"
            "查偏好用 interest；查既有流程用 procedure。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "intent": {
                    "type": "string",
                    "enum": ["answer", "timeline", "interest", "procedure"],
                    "default": "answer",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 8},
                "time_start": {"type": "string", "format": "date-time"},
                "time_end": {"type": "string", "format": "date-time"},
                "time_filter": {
                    "type": "string",
                    "description": "today、yesterday、recent_3d/7d/30d、YYYY-MM-DD 或日期范围",
                },
                "memory_kind": {
                    "type": "string",
                    "enum": ["event", "profile", "preference", "procedure"],
                },
                "memory_kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["event", "profile", "preference", "procedure"],
                    },
                    "uniqueItems": True,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=recall_memory,
    )


def build_memorize_tool(memorizer: ExplicitMemorizer) -> Tool:
    async def memorize(
        *,
        summary: str,
        memory_kind: str = "event",
        source_ref: str = "",
        tool_requirement: str | None = None,
        steps: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, str]:
        context = current_memory_tool_context()
        channel = context.channel if context is not None else ""
        chat_id = context.chat_id if context is not None else ""
        stable_source = source_ref.strip() or _explicit_source_ref(
            summary,
            memory_kind,
            channel,
            chat_id,
            context.source_ref if context is not None else "",
        )
        extra = dict(metadata or {})
        if tool_requirement is not None:
            extra["tool_requirement"] = tool_requirement
        if steps is not None:
            extra["steps"] = steps
        result = await memorizer.remember(
            summary=summary,
            memory_kind=memory_kind,
            source_ref=stable_source,
            scope_channel=channel,
            scope_chat_id=chat_id,
            extra=extra,
            assert_current=context.assert_current if context is not None else None,
            fenced_write=context.fenced_write if context is not None else None,
        )
        return {
            "item_id": str(result.item_id),
            "status": str(result.status),
            "actual_kind": str(result.actual_kind),
        }

    return Tool(
        name="memorize",
        description="将用户明确要求记住的事实、偏好或可执行流程写入长期记忆。",
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1},
                "memory_kind": {
                    "type": "string",
                    "enum": ["event", "profile", "preference", "procedure"],
                    "default": "event",
                },
                "tool_requirement": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        handler=memorize,
    )


def build_forget_memory_tool(store: MemoryStore) -> Tool:
    async def forget_memory(*, ids: list[str]) -> dict[str, list[str]]:
        context = current_memory_tool_context()
        guard = context.assert_current if context and context.assert_current else (lambda: None)
        write_scope = context.fenced_write if context and context.fenced_write else nullcontext
        clean_ids = list(dict.fromkeys(value.strip() for value in ids if value.strip()))
        visible: list[str] = []
        for item_id in clean_ids:
            item = store.get_item(item_id)
            if item is None:
                continue
            extra = item.get("extra_json")
            item_scope = extra if isinstance(extra, dict) else {}
            item_channel = str(item_scope.get("scope_channel") or "")
            item_chat_id = str(item_scope.get("scope_chat_id") or "")
            if context is None or (
                (not item_channel and not item_chat_id)
                or (
                    (not context.channel or item_channel == context.channel)
                    and (not context.chat_id or item_chat_id == context.chat_id)
                )
            ):
                visible.append(item_id)
        known = visible
        missing = [item_id for item_id in clean_ids if item_id not in known]
        guard()
        with write_scope():
            store.mark_superseded_batch(
                tuple(known),
                scope_channel=context.channel if context else "",
                scope_chat_id=context.chat_id if context else "",
            )
        return {"superseded_ids": known, "missing_ids": missing}

    return Tool(
        name="forget_memory",
        description="将已确认错误或不再适用的长期记忆标记为失效；不会物理删除审计记录。",
        parameters={
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "需要失效的 memory item id 列表。",
                }
            },
            "required": ["ids"],
            "additionalProperties": False,
        },
        handler=forget_memory,
    )


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _parse_time_filter(value: str) -> tuple[datetime, datetime] | None:
    text = value.strip()
    if not text:
        return None
    now = datetime.now(_LOCAL_TZ)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text == "today":
        return today, today + timedelta(days=1)
    if text == "yesterday":
        return today - timedelta(days=1), today
    if text in _RECENT_PRESETS:
        return now - timedelta(days=_RECENT_PRESETS[text]), now
    if "~" in text:
        left, right = (part.strip() for part in text.split("~", 1))
        start, end = _parse_day(left), _parse_day(right)
        return None if start is None or end is None else (start, end + timedelta(days=1))
    day = _parse_day(text)
    return None if day is None else (day, day + timedelta(days=1))


def _parse_day(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=_LOCAL_TZ)
    except ValueError:
        return None


def _explicit_source_ref(
    summary: str,
    kind: str,
    channel: str,
    chat_id: str,
    turn_source_ref: str,
) -> str:
    digest = hashlib.sha256(
        f"{turn_source_ref}\0{channel}\0{chat_id}\0{kind}\0{summary.strip()}".encode()
    ).hexdigest()
    return f"explicit:{digest}"


__all__ = ["build_forget_memory_tool", "build_memorize_tool", "build_recall_memory_tool"]
