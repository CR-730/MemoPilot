"""MCP Server 配置、连接和工具注册的原子生命周期管理。"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from memopilot.extensions.mcp import McpServerClient, McpServerConfig
from memopilot.runtime.tools import Tool, ToolRegistry


class McpClient(Protocol):
    config: McpServerConfig

    async def as_tools(self) -> tuple[Tool, ...]: ...

    async def close(self) -> None: ...


McpClientFactory = Callable[[McpServerConfig], McpClient]


class McpServerRegistry:
    """管理多个官方 SDK stdio Actor，并同步其工具与持久化配置。"""

    def __init__(
        self,
        config_path: Path,
        tool_registry: ToolRegistry,
        *,
        client_factory: McpClientFactory = McpServerClient,
        on_tools_changed: Callable[[], None] | None = None,
    ) -> None:
        self._config_path = config_path
        self._tool_registry = tool_registry
        self._client_factory = client_factory
        self._on_tools_changed = on_tools_changed
        self._configs: dict[str, McpServerConfig] = {}
        self._clients: dict[str, McpClient] = {}
        self._server_tools: dict[str, tuple[str, ...]] = {}
        self._unavailable: dict[str, str] = {}
        self._diagnostics: list[str] = []
        self._connect_task: asyncio.Task[None] | None = None
        self._connecting: set[McpClient] = set()
        self._mutation_lock = asyncio.Lock()
        self._closed = False

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    async def import_configs(
        self,
        configs: tuple[McpServerConfig, ...],
        *,
        only_if_missing: bool = True,
    ) -> None:
        """离线导入类型化配置；已有 JSON 默认作为事实源且不被覆盖。"""
        async with self._mutation_lock:
            if self._closed:
                raise RuntimeError("MCP Registry 已关闭")
            loaded, diagnostics = self._load_configs()
            self._diagnostics.extend(diagnostics)
            if self._config_path.exists() and only_if_missing:
                self._configs.update(loaded)
                return
            imported = dict(loaded)
            for config in configs:
                if config.server_id in imported:
                    continue
                # 调用公开校验边界，确保任何持久化 env 都只是 ${ENV} 引用。
                config.environment_references()
                imported[config.server_id] = config
            if imported != loaded or not self._config_path.exists():
                self._save_configs(imported)
            self._configs = imported

    async def load_and_connect_all(self) -> None:
        """加载全部配置；每个 Server 独立连接，单点失败不会阻塞其他项。"""
        async with self._mutation_lock:
            if self._closed:
                return
            configs, diagnostics = self._load_configs()
            self._diagnostics.extend(diagnostics)
            self._configs.update(configs)

            for config in configs.values():
                if self._closed:
                    return
                if config.server_id in self._clients:
                    continue
                if not config.enabled:
                    self._unavailable[config.server_id] = "disabled"
                    continue
                try:
                    await self._connect_and_register(config)
                    self._notify_tools_changed()
                except asyncio.CancelledError:
                    raise
                except McpRegistryClosedError:
                    return
                except Exception as exc:
                    message = (
                        f"MCP Server {config.server_id!r} 启动失败: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._unavailable[config.server_id] = message
                    self._diagnostics.append(message)

    def start_connect_all_background(self) -> None:
        """后台执行启动重连，供 Bootstrap 在不阻塞核心服务时使用。"""
        if self._closed:
            raise RuntimeError("MCP Registry 已关闭")
        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(
                self.load_and_connect_all(), name="mcp-connect-all"
            )

    async def add(
        self,
        name: str,
        command: list[str],
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """连接成功并完整注册工具后才发布 Server；失败不留下半状态。"""
        try:
            config = McpServerConfig(
                server_id=name,
                command=tuple(command),
                args=tuple(args or ()),
                env=dict(env or {}),
                cwd=_resolve_cwd(cwd),
            )
        except (TypeError, ValueError) as exc:
            return f"MCP server {name!r} 配置无效：{exc}"
        async with self._mutation_lock:
            if self._closed:
                return "MCP Registry 已关闭。"
            if name in self._configs or name in self._clients:
                return f"MCP server {name!r} 已存在。如需更新，请先 mcp_remove。"
            try:
                client, tool_names = await self._connect_and_register(config)
            except McpRegistryClosedError:
                return "MCP Registry 已关闭。"
            except McpToolConflictError as exc:
                return f"连接 MCP server {name!r} 失败：工具冲突：{exc}"
            except Exception as exc:
                return f"连接 MCP server {name!r} 失败：{type(exc).__name__}: {exc}"

            self._configs[name] = config
            self._unavailable.pop(name, None)
            try:
                self._save_configs(self._configs)
            except Exception as exc:
                self._configs.pop(name, None)
                for tool_name in tool_names:
                    self._tool_registry.unregister(tool_name)
                self._server_tools.pop(name, None)
                self._clients.pop(name, None)
                await client.close()
                return f"保存 MCP server {name!r} 配置失败：{type(exc).__name__}: {exc}"
            self._notify_tools_changed()
            rendered = "\n".join(f"- {tool_name}" for tool_name in tool_names)
            return (
                f"已连接 MCP server {name!r}，注册了 {len(tool_names)} 个工具："
                + (f"\n{rendered}" if rendered else " 无")
            )

    async def remove(self, name: str) -> str:
        async with self._mutation_lock:
            if self._closed:
                return "MCP Registry 已关闭。"
            if name not in self._configs and name not in self._clients:
                names = sorted(set(self._configs) | set(self._clients))
                return f"MCP server {name!r} 不存在，当前已注册：{names or '无'}"
            updated = dict(self._configs)
            updated.pop(name, None)
            try:
                self._save_configs(updated)
            except Exception as exc:
                return f"保存 MCP server {name!r} 删除结果失败：{type(exc).__name__}: {exc}"
            self._configs = updated
            self._unavailable.pop(name, None)
            for tool_name in self._server_tools.pop(name, ()):
                self._tool_registry.unregister(tool_name)
            client = self._clients.pop(name, None)
            if client is not None:
                try:
                    await client.close()
                except Exception as exc:
                    self._diagnostics.append(
                        f"MCP Server {name!r} 关闭失败: {type(exc).__name__}: {exc}"
                    )
            self._notify_tools_changed()
            return f"已注销 MCP server {name!r}。"

    def list_servers(self) -> tuple[dict[str, object], ...]:
        """返回适合工具输出的脱敏、稳定排序视图。"""
        return tuple(
            {
                "name": name,
                "status": "connected" if name in self._clients else "unavailable",
                "tools": list(self._server_tools.get(name, ())),
                "config": config.public_view(),
                "diagnostic": self._unavailable.get(name),
            }
            for name, config in sorted(self._configs.items())
        )

    async def shutdown(self) -> None:
        self._closed = True
        task = self._connect_task
        self._connect_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        async with self._mutation_lock:
            connecting = tuple(self._connecting)
            clients = tuple(self._clients.values())
            self._connecting.clear()
            self._clients.clear()
            for tool_names in self._server_tools.values():
                for tool_name in tool_names:
                    self._tool_registry.unregister(tool_name)
            self._server_tools.clear()
            self._notify_tools_changed()
        await asyncio.gather(
            *(client.close() for client in (*connecting, *clients)),
            return_exceptions=True,
        )

    async def _connect_and_register(
        self, config: McpServerConfig
    ) -> tuple[McpClient, tuple[str, ...]]:
        if self._closed:
            raise McpRegistryClosedError("MCP Registry 已关闭")
        client = self._client_factory(config)
        self._connecting.add(client)
        try:
            tools = await client.as_tools()
            if self._closed:
                raise McpRegistryClosedError("MCP Registry 已关闭")
            try:
                self._tool_registry.register_many(
                    tools,
                    risk="external-side-effect",
                    source_type="mcp",
                    source_name=config.server_id,
                )
            except ValueError as exc:
                raise McpToolConflictError(str(exc)) from exc
        except BaseException:
            close_task = asyncio.create_task(client.close())
            try:
                await asyncio.shield(close_task)
            finally:
                if close_task.done():
                    self._connecting.discard(client)
            raise
        self._connecting.discard(client)
        tool_names = tuple(tool.name for tool in tools)
        self._clients[config.server_id] = client
        self._server_tools[config.server_id] = tool_names
        self._unavailable.pop(config.server_id, None)
        return client, tool_names

    def _notify_tools_changed(self) -> None:
        if self._on_tools_changed is not None:
            self._on_tools_changed()

    def _load_configs(self) -> tuple[dict[str, McpServerConfig], list[str]]:
        if not self._config_path.exists():
            return {}, []
        diagnostics: list[str] = []
        try:
            payload = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, [f"读取 MCP 配置失败: {type(exc).__name__}: {exc}"]
        raw_servers = payload.get("servers", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw_servers, dict):
            return {}, ["读取 MCP 配置失败: servers 必须是对象"]
        configs: dict[str, McpServerConfig] = {}
        for name, raw in sorted(raw_servers.items()):
            try:
                if not isinstance(raw, dict):
                    raise ValueError("配置必须是对象")
                command = raw.get("command")
                args = raw.get("args", [])
                env = raw.get("env", {})
                if not isinstance(command, list) or not isinstance(args, list):
                    raise ValueError("command/args 必须是数组")
                if not isinstance(env, dict):
                    raise ValueError("env 必须是对象")
                configs[str(name)] = McpServerConfig(
                    server_id=str(name),
                    command=tuple(command),
                    args=tuple(args),
                    env={str(key): str(value) for key, value in env.items()},
                    cwd=Path(str(raw["cwd"])) if raw.get("cwd") else None,
                    enabled=bool(raw.get("enabled", True)),
                    startup_timeout_seconds=float(raw.get("startup_timeout_seconds", 15)),
                    call_timeout_seconds=float(raw.get("call_timeout_seconds", 30)),
                    shutdown_timeout_seconds=float(raw.get("shutdown_timeout_seconds", 5)),
                    max_restarts=int(raw.get("max_restarts", 3)),
                )
            except (TypeError, ValueError) as exc:
                diagnostics.append(f"MCP Server {name!r} 配置无效: {exc}")
        return configs, diagnostics

    def _save_configs(self, configs: dict[str, McpServerConfig]) -> None:
        payload = {
            "servers": {
                name: _config_to_storage(config)
                for name, config in sorted(configs.items())
            }
        }
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self._config_path.name}.",
            suffix=".tmp",
            dir=self._config_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._config_path)
        finally:
            temporary.unlink(missing_ok=True)


def _config_to_storage(config: McpServerConfig) -> dict[str, object]:
    return {
        "command": list(config.command),
        "args": list(config.args),
        "env": dict(sorted(config.environment_references().items())),
        "cwd": str(config.cwd) if config.cwd is not None else None,
        "enabled": config.enabled,
        "startup_timeout_seconds": config.startup_timeout_seconds,
        "call_timeout_seconds": config.call_timeout_seconds,
        "shutdown_timeout_seconds": config.shutdown_timeout_seconds,
        "max_restarts": config.max_restarts,
    }


class McpToolConflictError(RuntimeError):
    """远端工具别名与当前 Registry 已有工具冲突。"""


class McpRegistryClosedError(RuntimeError):
    """连接尚未发布时 Registry 已进入永久关闭状态。"""


def _resolve_cwd(cwd: str | None) -> Path | None:
    return Path(cwd).expanduser().resolve() if cwd else None


__all__ = [
    "McpClient",
    "McpClientFactory",
    "McpServerRegistry",
    "McpRegistryClosedError",
    "McpToolConflictError",
]
