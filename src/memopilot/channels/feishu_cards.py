"""飞书 schema 2.0 过程卡渲染。"""

from __future__ import annotations

import json
from dataclasses import dataclass

_THINKING_LIVE_TAIL = 1400
_TOOL_LIVE_TAIL = 1000
_REPLY_LIVE_TAIL = 1100
_TOOL_PREVIEW_LIMIT = 80
_THINKING_HEADER = "💭 思考过程"


@dataclass
class ToolLiveLine:
    call_id: str
    tool_name: str
    intent: str
    target: str
    status: str = "running"


def build_live_card(
    thinking: str,
    tool_lines: list[ToolLiveLine],
    reply: str,
) -> str:
    elements: list[dict[str, object]] = []
    thinking_body = _tail_text(thinking.strip(), _THINKING_LIVE_TAIL)
    if thinking_body:
        elements.append(_markdown(f"**{_THINKING_HEADER}**\n{thinking_body}"))
    if tool_lines:
        elements.append(
            _markdown(_tail_text(format_tool_live(tool_lines), _TOOL_LIVE_TAIL))
        )
    reply_body = _tail_text(reply.strip(), _REPLY_LIVE_TAIL)
    if reply_body:
        elements.append(_markdown(f"**临时回复**\n{reply_body}"))
    if not elements:
        elements.append(_markdown("正在思考…"))
    return _dump_card(elements)


def build_summary_card(thinking: str, tool_lines: list[ToolLiveLine]) -> str:
    elements: list[dict[str, object]] = []
    thinking_body = _tail_text(thinking.strip(), _THINKING_LIVE_TAIL)
    if thinking_body:
        elements.append(_thinking_panel(thinking_body, expanded=False))
    if tool_lines:
        elements.append(
            _markdown(_tail_text(format_tool_live(tool_lines), _TOOL_LIVE_TAIL))
        )
    if not elements:
        elements.append(_markdown("本轮已完成"))
    return _dump_card(elements)


def format_tool_live(lines: list[ToolLiveLine]) -> str:
    shown = lines[-12:]
    rows = ["**🔧 工具调用**"]
    hidden = len(lines) - len(shown)
    if hidden > 0:
        rows.append(f"... {hidden} more")
    for line in shown:
        status = "..."
        if line.status == "done":
            status = "✅"
        elif line.status == "error":
            status = "✗"
        target = f" {line.target}" if line.target else ""
        rows.append(
            f"{_tool_emoji(line.tool_name)} {_clip_inline(line.tool_name, 32)}: "
            f"{line.intent}{target} {status}"
        )
    if lines and all(line.status != "running" for line in lines):
        rows.append(f"Done · {len(lines)} tools")
    return "\n".join(rows)


def format_tool_intent(arguments: dict[str, object]) -> str:
    value = arguments.get("description")
    if value is None or value == "":
        return ""
    return _clip_inline(_stringify_tool_value(value), _TOOL_PREVIEW_LIMIT)


def format_tool_target(arguments: dict[str, object]) -> str:
    for key in (
        "cmd",
        "command",
        "query",
        "url",
        "path",
        "file",
        "text",
        "content",
        "prompt",
        "name",
    ):
        value = arguments.get(key)
        if value is not None and value != "":
            rendered = _clip_inline(_stringify_tool_value(value), _TOOL_PREVIEW_LIMIT)
            return f'"{rendered}"'
    return ""


def _dump_card(elements: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "schema": "2.0",
            "config": {"update_multi": True},
            "body": {"elements": elements},
        },
        ensure_ascii=False,
    )


def _markdown(content: str) -> dict[str, object]:
    return {"tag": "markdown", "content": content}


def _thinking_panel(content: str, *, expanded: bool) -> dict[str, object]:
    return {
        "tag": "collapsible_panel",
        "expanded": expanded,
        "header": {"title": {"tag": "plain_text", "content": _THINKING_HEADER}},
        "elements": [_markdown(content)],
    }


def _stringify_tool_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _clip_inline(text: str, limit: int) -> str:
    plain = " ".join(str(text).split())
    if len(plain) <= limit:
        return plain
    if limit <= 3:
        return plain[:limit]
    return plain[: limit - 3] + "..."


def _tail_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3) :]


def _tool_emoji(tool_name: str) -> str:
    name = tool_name.lower()
    if name.startswith("mcp"):
        return "📡"
    if "search" in name:
        return "🔍"
    if "web" in name or "url" in name:
        return "🌐"
    if "file" in name or "read" in name:
        return "📄"
    if "write" in name or "save" in name:
        return "💾"
    if "shell" in name or "exec" in name:
        return "⚙"
    return "🔧"


__all__ = [
    "ToolLiveLine",
    "build_live_card",
    "build_summary_card",
    "format_tool_intent",
    "format_tool_target",
]
