"""Drift 专用工具注册表与状态收尾。

这是旧原型 ``DriftTurnPipeline`` 的最小可运行适配：
本轮使用独立 ToolRegistry；``message_push`` 成功后由 Runtime Observer
把可见工具收缩为 write/edit/finish；``finish_drift`` 负责校验并保存结果。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memopilot.runtime.tools import Tool, ToolRegistry


@dataclass
class DriftRunState:
    skill_names: frozenset[str]
    message_sent: bool = False
    finished: bool = False
    finish_payload: dict[str, str] | None = None


SendDriftMessage = Callable[[str, list[str]], Awaitable[bool]]
SaveDriftFinish = Callable[[dict[str, str]], Awaitable[None] | None]


def build_drift_tool_registry(
    *,
    workspace: Path,
    builtin_skills_dir: Path | None = None,
    state: DriftRunState,
    send_message: SendDriftMessage | None = None,
    save_finish: SaveDriftFinish | None = None,
    shared_tools: ToolRegistry | None = None,
    connected_servers: frozenset[str] = frozenset(),
) -> ToolRegistry:
    workspace = workspace.resolve()
    builtin_skills_dir = (
        builtin_skills_dir.resolve() if builtin_skills_dir is not None else None
    )

    async def read_file(path: str) -> str:
        target = _safe_path(workspace, path)
        if (
            not target.exists()
            and builtin_skills_dir is not None
            and Path(path).as_posix().startswith("skills/")
        ):
            relative = Path(path).as_posix().removeprefix("skills/")
            builtin_target = _safe_path(builtin_skills_dir, relative)
            if builtin_target.exists():
                target = builtin_target
        return target.read_text(encoding="utf-8")

    async def write_file(path: str, content: str) -> str:
        target = _safe_path(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": str(target.relative_to(workspace))})

    async def edit_file(path: str, old: str, new: str) -> str:
        target = _safe_path(workspace, path)
        current = target.read_text(encoding="utf-8")
        if old not in current:
            return json.dumps({"error": "old text not found"}, ensure_ascii=False)
        target.write_text(current.replace(old, new, 1), encoding="utf-8")
        return json.dumps({"ok": True, "path": str(target.relative_to(workspace))})

    async def message_push(
        message: str = "",
        image: str = "",
        media: list[str] | None = None,
        **_: Any,
    ) -> str:
        del image
        if state.message_sent:
            return json.dumps({"error": "message_push already used"}, ensure_ascii=False)
        text = message.strip()
        media_paths = [str(item).strip() for item in (media or []) if str(item).strip()]
        if not text and not media_paths:
            return json.dumps({"error": "message or media is required"}, ensure_ascii=False)
        if send_message is None or not await send_message(text, media_paths):
            return json.dumps({"error": "message_push failed"}, ensure_ascii=False)
        state.message_sent = True
        return json.dumps({"ok": True}, ensure_ascii=False)

    async def finish_drift(
        skill_used: str,
        one_line: str,
        next: str,
        message_result: str,
        note: str = "",
        **_: Any,
    ) -> str:
        skill = skill_used.strip()
        summary = one_line.strip()
        next_action = next.strip()
        if skill not in state.skill_names:
            return json.dumps({"error": f"unknown skill: {skill}"}, ensure_ascii=False)
        if not summary or not next_action:
            return json.dumps({"error": "one_line and next are required"}, ensure_ascii=False)
        if message_result not in {"sent", "silent"}:
            return json.dumps(
                {"error": "message_result must be sent or silent"},
                ensure_ascii=False,
            )
        if message_result == "sent" and not state.message_sent:
            return json.dumps(
                {"error": "message_result=sent requires successful message_push"},
                ensure_ascii=False,
            )
        if message_result == "silent" and state.message_sent:
            return json.dumps(
                {"error": "message_result=silent conflicts with message_push"},
                ensure_ascii=False,
            )
        payload = {
            "skill_used": skill,
            "one_line": summary,
            "next": next_action,
            "message_result": message_result,
            "note": note.strip(),
        }
        if save_finish is not None:
            result = save_finish(payload)
            if result is not None:
                await result
        state.finish_payload = payload
        state.finished = True
        return json.dumps({"ok": True}, ensure_ascii=False)

    mounted_servers: set[str] = set()

    async def mount_server(server: str) -> str:
        server_id = server.strip()
        if server_id not in connected_servers:
            return json.dumps(
                {"error": f"MCP server unavailable: {server_id}"},
                ensure_ascii=False,
            )
        if shared_tools is None:
            return json.dumps({"error": "shared MCP registry unavailable"}, ensure_ascii=False)
        names = [
            name
            for name in shared_tools.tool_names
            if (document := shared_tools.get_document(name)) is not None
            and document.source_type == "mcp"
            and document.source_name == server_id
        ]
        registry.register_existing(shared_tools, names)
        mounted_servers.add(server_id)
        return json.dumps(
            {"ok": True, "server": server_id, "tools": names},
            ensure_ascii=False,
        )

    registry = ToolRegistry(
        (
            Tool(
                "read_file",
                "读取 Drift 工作区中的文件。",
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                read_file,
            ),
            Tool(
                "write_file",
                "写入 Drift 工作区文件。",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                write_file,
                timeout_seconds=30,
            ),
            Tool(
                "edit_file",
                "替换 Drift 工作区文件中的一段文本。",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                },
                edit_file,
                timeout_seconds=30,
            ),
            Tool(
                "message_push",
                "向用户发送一条 Drift 消息，每轮最多一次。",
                {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "image": {"type": "string"},
                        "media": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["message"],
                },
                message_push,
            ),
            Tool(
                "finish_drift",
                "保存 Drift 状态并结束本轮。",
                {
                    "type": "object",
                    "properties": {
                        "skill_used": {"type": "string"},
                        "one_line": {"type": "string"},
                        "next": {"type": "string"},
                        "message_result": {"type": "string", "enum": ["sent", "silent"]},
                        "note": {"type": "string"},
                    },
                    "required": ["skill_used", "one_line", "next", "message_result"],
                },
                finish_drift,
            ),
            Tool(
                "mount_server",
                "挂载一个已连接的 MCP Server，使其工具在本次 Drift 中可用。",
                {
                    "type": "object",
                    "properties": {"server": {"type": "string"}},
                    "required": ["server"],
                },
                mount_server,
            ),
        )
    )
    if shared_tools is not None:
        # Drift 的本地文件与发送工具保留专用实现；其余公共工具直接复用
        # 主 Agent 注册表，确保提示词声明与实际可见工具一致。
        shared_names = tuple(
            name
            for name in (
                "recall_memory",
                "shell",
                "task_output",
                "task_stop",
                "web_search",
                "web_fetch",
                "fetch_messages",
                "search_messages",
            )
            if name not in registry.tool_names and shared_tools.get_tool(name) is not None
        )
        registry.register_existing(shared_tools, shared_names)
    return registry


def _safe_path(workspace: Path, raw: str) -> Path:
    candidate = (workspace / raw).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("Drift 文件路径不能越出工作区") from exc
    return candidate


__all__ = ["DriftRunState", "build_drift_tool_registry"]
