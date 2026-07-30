"""MemoPilot 内置公共工具的统一构造与注册。

工具本体保留旧实现；本文件只负责把旧的 ``execute`` 合同接到当前
``Tool`` 数据类，并把当前 SQLite 消息表适配为旧工具需要的查询接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memopilot.runtime.tools import Tool, ToolRegistry

from .filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    ToolResult,
    WriteFileTool,
)
from .http import HttpRequester
from .message_lookup import FetchMessagesTool, SearchMessagesTool
from .message_push import MessagePushTool
from .shell import ShellTaskOutputTool, ShellTaskStopTool, ShellTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool


class OperationalMessageStoreAdapter:
    """把当前 operational.db 的 messages 表适配为旧查询工具合同。"""

    def __init__(self, repository: Any) -> None:
        self._repository = repository

    def _connect(self) -> Any:
        return self._repository._connect()  # noqa: SLF001 - 适配既有仓储连接

    def fetch_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        connection = self._connect()
        try:
            placeholders = ",".join("?" for _ in ids)
            rows = connection.execute(
                f"""
                SELECT message_id, session_key, role, content,
                       session_position, created_at
                FROM messages
                WHERE message_id IN ({placeholders})
                """,
                tuple(ids),
            ).fetchall()
        finally:
            connection.close()
        by_id = {str(row["message_id"]): self._row(row, False) for row in rows}
        return [by_id[item_id] for item_id in ids if item_id in by_id]

    def fetch_by_ids_with_context(
        self,
        ids: list[str],
        context: int,
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        connection = self._connect()
        try:
            placeholders = ",".join("?" for _ in ids)
            targets = connection.execute(
                f"""
                SELECT message_id, session_key, session_position
                FROM messages
                WHERE message_id IN ({placeholders})
                """,
                tuple(ids),
            ).fetchall()
            result: dict[str, dict[str, Any]] = {}
            for target in targets:
                session_key = str(target["session_key"])
                position = int(target["session_position"])
                rows = connection.execute(
                    """
                    SELECT message_id, session_key, role, content,
                           session_position, created_at
                    FROM messages
                    WHERE session_key = ?
                      AND session_position BETWEEN ? AND ?
                    ORDER BY session_position
                    """,
                    (session_key, position - context, position + context),
                ).fetchall()
                target_id = str(target["message_id"])
                for row in rows:
                    message = self._row(
                        row,
                        str(row["message_id"]) == target_id,
                    )
                    result[str(row["message_id"])] = message
        finally:
            connection.close()
        ordered = sorted(
            result.values(),
            key=lambda item: (
                str(item.get("session_key", "")),
                int(item.get("seq", 0)),
            ),
        )
        return ordered

    def search_messages(
        self,
        term: str,
        *,
        session_key: str | None,
        role: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses = ["content LIKE ?"]
        params: list[Any] = [f"%{term}%"]
        if session_key:
            clauses.append("session_key = ?")
            params.append(session_key)
        if role:
            clauses.append("role = ?")
            params.append(role)
        where = " AND ".join(clauses)
        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) AS count FROM messages WHERE {where}",
                    tuple(params),
                ).fetchone()["count"]
            )
            rows = connection.execute(
                f"""
                SELECT message_id, session_key, role, content,
                       session_position, created_at
                FROM messages
                WHERE {where}
                ORDER BY created_at, session_position
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        finally:
            connection.close()
        return [self._row(row, False) for row in rows], total

    @staticmethod
    def _row(row: Any, in_source_ref: bool) -> dict[str, Any]:
        return {
            "id": str(row["message_id"]),
            "session_key": str(row["session_key"]),
            "seq": int(row["session_position"] or 0),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "timestamp": str(row["created_at"]),
            "in_source_ref": in_source_ref,
        }


def _build_tool(implementation: Any) -> Tool:
    async def handler(**arguments: Any) -> Any:
        result = await implementation.execute(**arguments)
        if isinstance(result, ToolResult):
            if result.content_blocks:
                return {
                    "text": result.text or "工具执行完成。",
                    "content_blocks": result.content_blocks,
                }
            return result.text
        return result

    return Tool(
        name=implementation.name,
        description=implementation.description,
        parameters=implementation.parameters,
        handler=handler,
        timeout_seconds=30,
        source="builtin:common",
    )


def build_common_tools(
    *,
    workspace: Path,
    repository: Any,
    http_requester: HttpRequester | None = None,
    push_tool: MessagePushTool | None = None,
    multimodal: bool = True,
    vl_available: bool = False,
) -> tuple[Tool, ...]:
    """构建旧公共工具集的完整固定集合。"""
    readonly = (
        ReadFileTool(
            multimodal=multimodal,
            vl_available=vl_available,
        ),
        ListDirTool(),
        WebFetchTool(http_requester),
        WebSearchTool(),
    )
    message_store = OperationalMessageStoreAdapter(repository)
    implementations = (
        ShellTool(),
        ShellTaskOutputTool(),
        ShellTaskStopTool(),
        readonly[2],
        readonly[3],
        readonly[0],
        readonly[1],
        FetchMessagesTool(message_store),
        SearchMessagesTool(message_store),
        push_tool or MessagePushTool(),
        WriteFileTool(),
        EditFileTool(),
    )
    return tuple(_build_tool(tool) for tool in implementations)


def register_common_tools(
    registry: ToolRegistry,
    *,
    workspace: Path,
    repository: Any,
    http_requester: HttpRequester | None = None,
    push_tool: MessagePushTool | None = None,
    multimodal: bool = True,
    vl_available: bool = False,
) -> tuple[Tool, ...]:
    tools = build_common_tools(
        workspace=workspace,
        repository=repository,
        http_requester=http_requester,
        push_tool=push_tool,
        multimodal=multimodal,
        vl_available=vl_available,
    )
    search_hints = {
        "shell": "终端 脚本 bash 命令",
        "task_output": "后台任务输出 task_output 进程日志",
        "task_stop": "停止后台任务 task_stop 杀进程",
        "web_search": "谷歌 Bing 查资料",
        "web_fetch": "读取网址 浏览网页",
        "list_dir": "ls 查看目录",
        "fetch_messages": "消息回溯 按ID查对话原文 source_ref",
        "search_messages": "你之前说 聊过什么 历史对话",
    }
    for tool in tools:
        registry.register(
            tool,
            always_on=True,
            risk=(
                "external-side-effect"
                if tool.name in {"shell", "task_stop", "message_push"}
                else "write"
                if tool.name in {"write_file", "edit_file"}
                else "read-only"
            ),
            search_hint=search_hints.get(tool.name),
        )
    return tools


__all__ = [
    "OperationalMessageStoreAdapter",
    "build_common_tools",
    "register_common_tools",
]
