from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from memopilot.extensions.mcp import (
    McpInvocationError,
    McpServerClient,
    McpServerConfig,
)
from memopilot.extensions.mcp_registry import _config_to_storage
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import ToolRegistry

FAKE_SERVER = Path(__file__).parents[2] / "fixtures" / "fake_mcp_server.py"


def _config(**overrides: object) -> McpServerConfig:
    values: dict[str, object] = {
        "server_id": "fake",
        "command": (sys.executable,),
        "args": (str(FAKE_SERVER),),
        "startup_timeout_seconds": 5,
        "call_timeout_seconds": 1,
    }
    values.update(overrides)
    return McpServerConfig(**values)  # type: ignore[arg-type]


def test_mcp_config_rejects_shell_string_and_redacts_environment() -> None:
    with pytest.raises(ValueError, match="command 必须是字符串数组"):
        McpServerConfig(server_id="bad", command="python server.py")  # type: ignore[arg-type]

    config = _config(env={"TOKEN": "${TEST_MCP_TOKEN}", "MODE": "${TEST_MODE}"})
    assert config.public_view()["env"] == {
        "MODE": "${TEST_MODE}",
        "TOKEN": "${TEST_MCP_TOKEN}",
    }

    with pytest.raises(ValueError, match="所有 env 值.*引用"):
        _config(env={"API_KEY": "plain-secret"})
    with pytest.raises(ValueError, match="所有 env 值.*引用"):
        _config(env={"MODE": "test"})


def test_mcp_config_defensively_copies_mutable_mappings() -> None:
    environment = {"TOKEN": "${TEST_MCP_TOKEN}"}

    config = _config(env=environment)
    environment["TOKEN"] = "plain-secret"

    assert dict(config.env) == {"TOKEN": "${TEST_MCP_TOKEN}"}
    assert "tool_side_effects" not in config.public_view()
    assert not hasattr(config, "tool_side_effects")
    with pytest.raises(TypeError):
        config.env["TOKEN"] = "plain-secret"  # type: ignore[index]


@pytest.mark.parametrize("boundary", ["resolve", "public", "storage"])
def test_mcp_config_revalidates_environment_at_every_output_boundary(
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(env={"TOKEN": "${TEST_MCP_TOKEN}"})
    object.__setattr__(config, "env", {"TOKEN": "do-not-leak-this-secret"})
    monkeypatch.setenv("TEST_MCP_TOKEN", "resolved-secret")

    with pytest.raises(ValueError) as exc_info:
        if boundary == "resolve":
            config.resolved_environment()
        elif boundary == "public":
            config.public_view()
        else:
            _config_to_storage(config)

    assert "do-not-leak-this-secret" not in str(exc_info.value)


async def test_startup_reconnects_consume_budget_then_become_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memopilot.extensions.mcp as mcp_module

    attempts = 0

    @asynccontextmanager
    async def failing_stdio(*args: Any, **kwargs: Any) -> AsyncIterator[tuple[None, None]]:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("offline")
        yield None, None

    monkeypatch.setattr(mcp_module, "stdio_client", failing_stdio)
    client = McpServerClient(_config(max_restarts=1))
    try:
        with pytest.raises(McpInvocationError) as exc_info:
            await asyncio.wait_for(client.start(), timeout=0.5)
        assert exc_info.value.error_type == "mcp_unavailable"
        assert attempts == 2

        with pytest.raises(McpInvocationError) as later:
            await asyncio.wait_for(client.start(), timeout=0.05)
        assert later.value is exc_info.value
        assert attempts == 2
    finally:
        await client.close()


async def test_close_while_startup_is_blocked_settles_ready_and_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_run(self: McpServerClient) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(McpServerClient, "_run", blocked_run)
    client = McpServerClient(_config(shutdown_timeout_seconds=0.05))
    starter = asyncio.create_task(client.start())
    await entered.wait()

    await asyncio.wait_for(client.close(), timeout=0.2)

    with pytest.raises(McpInvocationError) as exc_info:
        await asyncio.wait_for(starter, timeout=0.1)
    assert exc_info.value.error_type == "mcp_shutdown"
    assert client._runner is None


async def test_official_stdio_client_lists_calls_and_maps_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MCP_TOKEN", "secret")
    client = McpServerClient(_config(env={"TOKEN": "${TEST_MCP_TOKEN}"}))
    try:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} >= {"echo", "fail", "serialized", "wait"}
        registry = ToolRegistry(await client.as_tools())
        observation = await registry.execute(
            FunctionCall("c1", "mcp_fake__echo", {"text": "hello"})
        )
        assert observation.ok is True
        assert observation.result["structured"] == {"text": "hello"}
    finally:
        await client.close()


async def test_is_error_becomes_structured_tool_failure() -> None:
    client = McpServerClient(_config())
    try:
        registry = ToolRegistry(await client.as_tools())
        observation = await registry.execute(FunctionCall("c1", "mcp_fake__fail", {}))
        assert observation.ok is False
        assert observation.error_type == "mcp_tool_error"
        assert "remote rejected" in (observation.error_message or "")
        assert observation.retryable is False
        assert not hasattr(observation, "side_effect_status")
    finally:
        await client.close()


async def test_calls_are_serialized_per_server() -> None:
    client = McpServerClient(_config())
    try:
        first, second = await asyncio.gather(
            client.call_tool("serialized", {"delay": 0.05}),
            client.call_tool("serialized", {"delay": 0.05}),
        )
        assert first.structured == {"max_active": 1}
        assert second.structured == {"max_active": 1}
    finally:
        await client.close()


async def test_timeout_is_reported_and_client_can_reconnect() -> None:
    client = McpServerClient(_config(call_timeout_seconds=0.05))
    try:
        with pytest.raises(McpInvocationError) as exc_info:
            await client.call_tool("wait", {"seconds": 0.2})
        assert exc_info.value.error_type == "mcp_timeout"

        result = await client.call_tool("echo", {"text": "after-timeout"})
        assert result.structured == {"text": "after-timeout"}
    finally:
        await client.close()


async def test_restart_exhaustion_fails_future_calls_without_hanging() -> None:
    client = McpServerClient(_config(call_timeout_seconds=0.05, max_restarts=0))
    try:
        with pytest.raises(McpInvocationError):
            await client.call_tool("wait", {"seconds": 0.2})
        with pytest.raises(McpInvocationError, match="重启次数超限"):
            await asyncio.wait_for(
                client.call_tool("echo", {"text": "never"}),
                timeout=0.2,
            )
    finally:
        await client.close()


async def test_cancelled_caller_does_not_crash_actor() -> None:
    client = McpServerClient(_config())
    try:
        cancelled = asyncio.create_task(client.call_tool("wait", {"seconds": 0.1}))
        await asyncio.sleep(0.02)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        result = await client.call_tool("echo", {"text": "still-alive"})
        assert result.structured == {"text": "still-alive"}
    finally:
        await client.close()


async def test_cancelled_queued_call_never_reaches_server() -> None:
    client = McpServerClient(_config())
    try:
        blocker = asyncio.create_task(client.call_tool("wait", {"seconds": 0.15}))
        await asyncio.sleep(0.02)
        queued = asyncio.create_task(client.call_tool("counted", {"label": "cancelled"}))
        await asyncio.sleep(0.02)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        await blocker

        count = await client.call_tool("get_count", {"label": "cancelled"})
        assert count.structured == {"result": 0}
    finally:
        await client.close()
