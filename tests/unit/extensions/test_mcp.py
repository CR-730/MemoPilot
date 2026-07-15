from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from memopilot.extensions.mcp import (
    McpInvocationError,
    McpServerClient,
    McpServerConfig,
)
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

    config = _config(env={"TOKEN": "${TEST_MCP_TOKEN}", "MODE": "test"})
    assert config.public_view()["env"] == {"MODE": "***", "TOKEN": "${TEST_MCP_TOKEN}"}


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
        assert observation.side_effect_status == "unknown"
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
