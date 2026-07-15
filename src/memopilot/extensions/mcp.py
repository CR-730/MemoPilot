"""基于官方 Python SDK v1 的单会话 MCP stdio 客户端。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ListToolsResult, TextContent

from memopilot.runtime.tools import Tool

_SERVER_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV_REFERENCE = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    server_id: str
    command: tuple[str, ...]
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    enabled: bool = True
    startup_timeout_seconds: float = 15.0
    call_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 5.0
    max_restarts: int = 3
    tool_side_effects: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SERVER_ID.fullmatch(self.server_id):
            raise ValueError(f"MCP server_id 格式无效: {self.server_id}")
        if not isinstance(self.command, tuple) or not self.command or any(
            not isinstance(item, str) or not item for item in self.command
        ):
            raise ValueError("MCP command 必须是字符串数组且不能为空")
        if not isinstance(self.args, tuple) or any(
            not isinstance(item, str) for item in self.args
        ):
            raise ValueError("MCP args 必须是字符串数组")
        if (
            self.startup_timeout_seconds <= 0
            or self.call_timeout_seconds <= 0
            or self.shutdown_timeout_seconds <= 0
        ):
            raise ValueError("MCP 启动和调用超时必须大于 0")
        if self.max_restarts < 0:
            raise ValueError("MCP max_restarts 不能小于 0")
        invalid_effects = set(self.tool_side_effects.values()).difference(
            {"none", "idempotent", "non_idempotent"}
        )
        if invalid_effects:
            raise ValueError("MCP tool_side_effects 包含无效等级")

    def resolved_environment(self) -> dict[str, str] | None:
        if not self.env:
            return None
        resolved: dict[str, str] = {}
        for key, value in self.env.items():
            match = _ENV_REFERENCE.fullmatch(value)
            if match is None:
                resolved[key] = value
                continue
            variable = match.group(1)
            if variable not in os.environ:
                raise ValueError(f"MCP 环境变量未配置: {variable}")
            resolved[key] = os.environ[variable]
        return resolved

    def public_view(self) -> dict[str, object]:
        return {
            "server_id": self.server_id,
            "command": list(self.command),
            "args": list(self.args),
            "env": {
                key: value if _ENV_REFERENCE.fullmatch(value) else "***"
                for key, value in sorted(self.env.items())
            },
            "cwd": str(self.cwd) if self.cwd is not None else None,
            "enabled": self.enabled,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "call_timeout_seconds": self.call_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "tool_side_effects": dict(sorted(self.tool_side_effects.items())),
        }


@dataclass(frozen=True, slots=True)
class McpRemoteTool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class McpCallResult:
    ok: bool = True
    content: tuple[dict[str, Any], ...] = ()
    structured: dict[str, Any] | None = None
    error_message: str | None = None


class McpInvocationError(RuntimeError):
    def __init__(self, error_type: str, message: str, *, retryable: bool = True) -> None:
        self.error_type = error_type
        self.retryable = retryable
        self.side_effect_status = "unknown"
        super().__init__(message)


@dataclass(slots=True)
class _Request:
    operation: Literal["list", "call", "close"]
    future: asyncio.Future[Any]
    name: str | None = None
    arguments: dict[str, Any] | None = None


class McpServerClient:
    """专用 Actor 持有 SDK 上下文，保证同 Server 请求串行且同任务清理。"""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[_Request] = asyncio.Queue()
        self._runner: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._closed = False
        self._terminal_error: McpInvocationError | None = None
        self._active_request: _Request | None = None
        self._errlog: TextIO = open(os.devnull, "w", encoding="utf-8")

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("MCP client 已关闭")
        if self._terminal_error is not None:
            raise self._terminal_error
        if not self.config.enabled:
            raise RuntimeError(f"MCP Server 已禁用: {self.config.server_id}")
        if self._runner is None:
            loop = asyncio.get_running_loop()
            self._ready = loop.create_future()
            self._runner = asyncio.create_task(
                self._run(),
                name=f"mcp-{self.config.server_id}",
            )
        assert self._ready is not None
        await asyncio.shield(self._ready)

    async def list_tools(self) -> tuple[McpRemoteTool, ...]:
        result = cast(ListToolsResult, await self._request("list"))
        return tuple(
            McpRemoteTool(tool.name, tool.description or "", dict(tool.inputSchema))
            for tool in result.tools
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> McpCallResult:
        result = cast(
            CallToolResult,
            await self._request("call", name=name, arguments=arguments),
        )
        content = tuple(item.model_dump(mode="json", by_alias=True) for item in result.content)
        error_message = None
        if result.isError:
            texts = [item.text for item in result.content if isinstance(item, TextContent)]
            error_message = "\n".join(texts) or "MCP Tool 返回 isError=true"
        return McpCallResult(
            ok=not result.isError,
            content=content,
            structured=result.structuredContent,
            error_message=error_message,
        )

    async def as_tools(self) -> tuple[Tool, ...]:
        tools = await self.list_tools()
        return tuple(self._map_tool(tool) for tool in tools)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runner is None:
            self._errlog.close()
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put(_Request("close", future))
        try:
            await asyncio.wait_for(
                asyncio.shield(self._runner),
                timeout=self.config.shutdown_timeout_seconds,
            )
        except TimeoutError:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        finally:
            self._runner = None
            self._errlog.close()

    def _map_tool(self, remote: McpRemoteTool) -> Tool:
        async def invoke(**arguments: Any) -> dict[str, Any]:
            result = await self.call_tool(remote.name, arguments)
            if not result.ok:
                raise McpInvocationError(
                    "mcp_tool_error",
                    result.error_message or "MCP Tool 执行失败",
                    retryable=False,
                )
            return {
                "content": result.content,
                "structured": result.structured,
            }

        return Tool(
            name=_tool_alias(self.config.server_id, remote.name),
            description=remote.description,
            parameters=remote.input_schema,
            handler=invoke,
            timeout_seconds=self.config.call_timeout_seconds + 1,
            source=f"mcp:{self.config.server_id}/{remote.name}",
            side_effect_class=self.config.tool_side_effects.get(
                remote.name,
                "non_idempotent",
            ),
        )

    async def _request(
        self,
        operation: Literal["list", "call"],
        *,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put(_Request(operation, future, name, arguments))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _run(self) -> None:
        restarts = 0
        while not self._closed:
            reconnect = False
            try:
                async with AsyncExitStack() as stack:
                    params = StdioServerParameters(
                        command=self.config.command[0],
                        args=[*self.config.command[1:], *self.config.args],
                        env=self.config.resolved_environment(),
                        cwd=self.config.cwd,
                    )
                    async with asyncio.timeout(self.config.startup_timeout_seconds):
                        read, write = await stack.enter_async_context(
                            stdio_client(params, errlog=self._errlog)
                        )
                        session = await stack.enter_async_context(ClientSession(read, write))
                        await session.initialize()
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(None)
                    while True:
                        request = await self._queue.get()
                        if request.operation != "close" and request.future.cancelled():
                            continue
                        self._active_request = request
                        if request.operation == "close":
                            if not request.future.done():
                                request.future.set_result(None)
                            self._active_request = None
                            return
                        try:
                            async with asyncio.timeout(self.config.call_timeout_seconds):
                                result: ListToolsResult | CallToolResult
                                if request.operation == "list":
                                    result = await session.list_tools()
                                else:
                                    assert request.name is not None
                                    result = await session.call_tool(
                                        request.name,
                                        request.arguments or {},
                                        read_timeout_seconds=timedelta(
                                            seconds=self.config.call_timeout_seconds
                                        ),
                                    )
                            if not request.future.done():
                                request.future.set_result(result)
                            self._active_request = None
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            if not request.future.done():
                                request.future.set_exception(_map_error(exc))
                            self._active_request = None
                            reconnect = True
                            break
            except asyncio.CancelledError:
                error = McpInvocationError(
                    "mcp_shutdown",
                    f"MCP Server 正在关闭: {self.config.server_id}",
                )
                if self._active_request is not None and not self._active_request.future.done():
                    self._active_request.future.set_exception(error)
                self._active_request = None
                self._fail_pending(error)
                raise
            except Exception as exc:
                mapped = _map_error(exc, startup=True)
                if self._ready is not None and not self._ready.done():
                    self._ready.set_exception(mapped)
                    return
                reconnect = True
            if reconnect:
                restarts += 1
                if restarts > self.config.max_restarts:
                    self._terminal_error = McpInvocationError(
                        "mcp_unavailable",
                        f"MCP Server 重启次数超限: {self.config.server_id}",
                    )
                    self._fail_pending(self._terminal_error)
                    return
                await asyncio.sleep(min(0.05 * (2 ** (restarts - 1)), 0.5))

    def _fail_pending(self, error: Exception) -> None:
        while not self._queue.empty():
            request = self._queue.get_nowait()
            if not request.future.done():
                request.future.set_exception(error)


def _map_error(exc: Exception, *, startup: bool = False) -> McpInvocationError:
    if isinstance(exc, McpInvocationError):
        return exc
    if isinstance(exc, TimeoutError):
        return McpInvocationError(
            "mcp_startup_timeout" if startup else "mcp_timeout",
            "MCP 启动超时" if startup else "MCP 工具调用超时",
        )
    return McpInvocationError(
        "mcp_startup_error" if startup else "mcp_protocol_error",
        f"{type(exc).__name__}: {exc}",
    )


def _tool_alias(server_id: str, remote_name: str) -> str:
    raw = f"mcp_{server_id}__{remote_name}"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if safe == raw and len(safe) <= 64:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:55]}_{digest}"


__all__ = [
    "McpCallResult",
    "McpInvocationError",
    "McpRemoteTool",
    "McpServerClient",
    "McpServerConfig",
]
