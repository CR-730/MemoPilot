from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from memopilot.extensions.mcp import McpServerConfig
from memopilot.extensions.mcp_manage_tools import (
    build_mcp_management_tools,
    register_mcp_management_tools,
)
from memopilot.extensions.mcp_registry import McpServerRegistry
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolRegistry


class FakeClient:
    def __init__(
        self,
        config: McpServerConfig,
        *,
        fail: bool = False,
        tool_name: str = "echo",
        wait: asyncio.Event | None = None,
    ) -> None:
        self.config = config
        self.fail = fail
        self.tool_name = tool_name
        self.wait = wait
        self.closed = False

    async def as_tools(self) -> tuple[Tool, ...]:
        if self.wait is not None:
            await self.wait.wait()
        if self.fail:
            raise RuntimeError("offline")

        async def invoke(**arguments: Any) -> dict[str, Any]:
            return arguments

        return (
            Tool(
                name=f"mcp_{self.config.server_id}__{self.tool_name}",
                description="fake",
                parameters={"type": "object", "properties": {}},
                handler=invoke,
                source=f"mcp:{self.config.server_id}/{self.tool_name}",
            ),
        )

    async def close(self) -> None:
        self.closed = True


def _config(server_id: str, **overrides: Any) -> McpServerConfig:
    values: dict[str, Any] = {
        "server_id": server_id,
        "command": ("python",),
        "args": ("server.py",),
    }
    values.update(overrides)
    return McpServerConfig(**values)


async def test_import_configs_is_offline_idempotent_and_keeps_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp.json"
    tools = ToolRegistry()
    created: list[FakeClient] = []

    def factory(config: McpServerConfig) -> FakeClient:
        client = FakeClient(config)
        created.append(client)
        return client

    registry = McpServerRegistry(path, tools, client_factory=factory)
    await registry.import_configs(
        (_config("docs", env={"TOKEN": "${DOCS_TOKEN}"}),)
    )
    await registry.import_configs((_config("docs"),))

    assert created == []
    assert tools.tool_names == ()
    assert [item["name"] for item in registry.list_servers()] == ["docs"]
    persisted = path.read_text(encoding="utf-8")
    assert "${DOCS_TOKEN}" in persisted

    path.write_text(
        json.dumps({"servers": {"local": {"command": ["python"]}}}),
        encoding="utf-8",
    )
    second = McpServerRegistry(path, tools, client_factory=factory)
    await second.import_configs((_config("settings"),), only_if_missing=True)
    assert [item["name"] for item in second.list_servers()] == ["local"]
    assert "settings" not in path.read_text(encoding="utf-8")


async def test_add_list_remove_persists_config_without_resolved_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCS_TOKEN", "real-secret")
    tools = ToolRegistry()
    clients: list[FakeClient] = []

    def factory(config: McpServerConfig) -> FakeClient:
        client = FakeClient(config)
        clients.append(client)
        return client

    path = tmp_path / "mcp_servers.json"
    changes: list[tuple[str, ...]] = []
    registry = McpServerRegistry(
        path,
        tools,
        client_factory=factory,
        on_tools_changed=lambda: changes.append(tools.tool_names),
    )
    result = await registry.add(
        "docs",
        ["python"],
        args=["server.py"],
        env={"DOCS_TOKEN": "${DOCS_TOKEN}", "MODE": "${DOCS_MODE}"},
        cwd=str(tmp_path),
    )

    assert "已连接" in result
    assert tools.tool_names == ("mcp_docs__echo",)
    document = tools.get_document("mcp_docs__echo")
    assert document is not None
    assert document.source_type == "mcp"
    assert document.source_name == "docs"
    assert tools.search("mcp_docs__echo")[0]["name"] == "mcp_docs__echo"
    listed = registry.list_servers()
    assert listed[0]["name"] == "docs"
    assert listed[0]["status"] == "connected"
    assert listed[0]["config"]["env"] == {
        "DOCS_TOKEN": "${DOCS_TOKEN}",
        "MODE": "${DOCS_MODE}",
    }
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["servers"]["docs"]["env"]["DOCS_TOKEN"] == "${DOCS_TOKEN}"
    assert "real-secret" not in path.read_text(encoding="utf-8")
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))

    assert "已注销" in await registry.remove("docs")
    assert tools.tool_names == ()
    assert tools.search("mcp_docs__echo") == []
    assert clients[0].closed is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"servers": {}}
    assert changes == [("mcp_docs__echo",), ()]


async def test_failed_or_conflicting_add_leaves_no_half_registered_state(
    tmp_path: Path,
) -> None:
    async def local_handler() -> str:
        return "local"

    tools = ToolRegistry(
        [
            Tool(
                name="mcp_docs__echo",
                description="local",
                parameters={"type": "object", "properties": {}},
                handler=local_handler,
            )
        ]
    )
    created: list[FakeClient] = []

    def factory(config: McpServerConfig) -> FakeClient:
        client = FakeClient(config, fail=config.server_id == "bad")
        created.append(client)
        return client

    path = tmp_path / "mcp.json"
    registry = McpServerRegistry(path, tools, client_factory=factory)
    assert "失败" in await registry.add("bad", ["python"])
    assert "冲突" in await registry.add("docs", ["python"])
    assert registry.list_servers() == ()
    assert tools.tool_names == ("mcp_docs__echo",)
    assert all(client.closed for client in created)
    assert not path.exists()


async def test_startup_failure_is_isolated_and_shutdown_cancels_background_connect(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "bad": {"command": ["python"]},
                    "good": {"command": ["python"]},
                }
            }
        ),
        encoding="utf-8",
    )
    wait = asyncio.Event()
    clients: list[FakeClient] = []

    def factory(config: McpServerConfig) -> FakeClient:
        client = FakeClient(
            config,
            fail=config.server_id == "bad",
            wait=wait if config.server_id == "good" else None,
        )
        clients.append(client)
        return client

    tools = ToolRegistry()
    registry = McpServerRegistry(path, tools, client_factory=factory)
    registry.start_connect_all_background()
    await asyncio.sleep(0)
    await registry.shutdown()

    assert tools.tool_names == ()
    assert all(client.closed for client in clients)

    # 同步启动时，单个失败不会阻塞其余 Server。
    wait.set()
    registry = McpServerRegistry(path, tools, client_factory=factory)
    await registry.load_and_connect_all()
    statuses = {item["name"]: item["status"] for item in registry.list_servers()}
    assert statuses == {"bad": "unavailable", "good": "connected"}
    assert tools.tool_names == ("mcp_good__echo",)
    assert any("bad" in item for item in registry.diagnostics)
    await registry.shutdown()
    assert tools.tool_names == ()


async def test_management_tools_delegate_with_current_tool_contract(tmp_path: Path) -> None:
    tools = ToolRegistry()
    registry = McpServerRegistry(
        tmp_path / "mcp.json",
        tools,
        client_factory=lambda config: FakeClient(config),
    )
    management = build_mcp_management_tools(registry)
    assert {tool.name for tool in management} == {"mcp_add", "mcp_remove", "mcp_list"}
    tool_registry = ToolRegistry()
    register_mcp_management_tools(tool_registry, registry)
    assert tool_registry.get_document("mcp_add").risk == "external-side-effect"  # type: ignore[union-attr]
    assert tool_registry.get_document("mcp_remove").risk == "write"  # type: ignore[union-attr]
    assert tool_registry.get_document("mcp_list").risk == "read-only"  # type: ignore[union-attr]

    added = await tool_registry.execute(
        FunctionCall(
            "1",
            "mcp_add",
            {
                "name": "docs",
                "command": ["python"],
                "args": ["server.py"],
                "env": {"MODE": "${DOCS_MODE}"},
                "cwd": str(tmp_path),
            },
        )
    )
    assert added.ok is True
    assert "已连接" in added.result
    listed = await tool_registry.execute(FunctionCall("2", "mcp_list", {}))
    assert listed.ok is True
    assert listed.result[0]["name"] == "docs"
    removed = await tool_registry.execute(
        FunctionCall("3", "mcp_remove", {"name": "docs"})
    )
    assert removed.ok is True


async def test_remote_mcp_tools_use_prototype_external_side_effect_risk(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry()
    registry = McpServerRegistry(
        tmp_path / "mcp.json",
        tools,
        client_factory=lambda config: FakeClient(config),
    )

    await registry.add("docs", ["python"])

    document = tools.get_document("mcp_docs__echo")
    assert document is not None
    assert document.risk == "external-side-effect"
    await registry.shutdown()


async def test_duplicate_add_has_stable_diagnostic(tmp_path: Path) -> None:
    registry = McpServerRegistry(
        tmp_path / "mcp.json",
        ToolRegistry(),
        client_factory=lambda config: FakeClient(config),
    )
    await registry.add("docs", ["python"])
    assert "已存在" in await registry.add("docs", ["python"])
    await registry.shutdown()


async def test_add_finishing_after_shutdown_cannot_publish_or_persist(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    clients: list[FakeClient] = []

    class BlockingClient(FakeClient):
        async def as_tools(self) -> tuple[Tool, ...]:
            entered.set()
            await release.wait()
            return await super().as_tools()

    def factory(config: McpServerConfig) -> FakeClient:
        client = BlockingClient(config)
        clients.append(client)
        return client

    path = tmp_path / "mcp.json"
    tools = ToolRegistry()
    registry = McpServerRegistry(path, tools, client_factory=factory)
    adding = asyncio.create_task(registry.add("docs", ["python"]))
    await entered.wait()
    shutting_down = asyncio.create_task(registry.shutdown())
    await asyncio.sleep(0)
    release.set()

    result = await adding
    await shutting_down

    assert "已关闭" in result
    assert tools.tool_names == ()
    assert not path.exists()
    assert clients[0].closed is True


async def test_concurrent_load_is_idempotent_and_shutdown_cannot_be_undone(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"servers": {"docs": {"command": ["python"]}}}),
        encoding="utf-8",
    )
    clients: list[FakeClient] = []

    def factory(config: McpServerConfig) -> FakeClient:
        client = FakeClient(config)
        clients.append(client)
        return client

    tools = ToolRegistry()
    registry = McpServerRegistry(path, tools, client_factory=factory)
    await asyncio.gather(registry.load_and_connect_all(), registry.load_and_connect_all())

    assert len(clients) == 1
    assert tools.tool_names == ("mcp_docs__echo",)
    await registry.shutdown()
    await registry.load_and_connect_all()
    assert len(clients) == 1
    assert tools.tool_names == ()
    before = path.read_text(encoding="utf-8")
    assert "已关闭" in await registry.remove("docs")
    assert path.read_text(encoding="utf-8") == before


async def test_management_tool_rejects_every_literal_environment_value(
    tmp_path: Path,
) -> None:
    registry = McpServerRegistry(
        tmp_path / "mcp.json",
        ToolRegistry(),
        client_factory=lambda config: FakeClient(config),
    )
    result = await registry.add("docs", ["python"], env={"MODE": "test"})
    assert "配置无效" in result
    assert "${ENV_NAME}" in result
    assert not (tmp_path / "mcp.json").exists()


async def test_persistence_failure_rolls_back_add_and_preserves_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = ToolRegistry()
    clients: list[FakeClient] = []

    def factory(config: McpServerConfig) -> FakeClient:
        client = FakeClient(config)
        clients.append(client)
        return client

    registry = McpServerRegistry(tmp_path / "mcp.json", tools, client_factory=factory)

    def fail_save(configs: dict[str, McpServerConfig]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(registry, "_save_configs", fail_save)
    assert "保存" in await registry.add("docs", ["python"])
    assert registry.list_servers() == ()
    assert tools.tool_names == ()
    assert clients[0].closed is True

    monkeypatch.undo()
    assert "已连接" in await registry.add("docs", ["python"])
    monkeypatch.setattr(registry, "_save_configs", fail_save)
    assert "删除结果失败" in await registry.remove("docs")
    assert registry.list_servers()[0]["status"] == "connected"
    assert tools.tool_names == ("mcp_docs__echo",)
    await registry.shutdown()
