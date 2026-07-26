from pathlib import Path

import pytest

from memopilot.proactive.drift_runtime import DriftRunState, build_drift_tool_registry
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolRegistry


@pytest.mark.asyncio
async def test_message_push_then_finish_drift_is_idempotently_guarded(tmp_path: Path) -> None:
    sent: list[str] = []

    async def send(text: str, media: list[str]) -> bool:
        del media
        sent.append(text)
        return True

    state = DriftRunState(frozenset({"alpha"}))
    registry = build_drift_tool_registry(
        workspace=tmp_path,
        state=state,
        send_message=send,
    )

    pushed = await registry.execute(
        FunctionCall("1", "message_push", {"message": "你好"})
    )
    assert pushed.ok and state.message_sent
    duplicate = await registry.execute(
        FunctionCall("2", "message_push", {"message": "重复"})
    )
    assert "already used" in str(duplicate.result)

    finished = await registry.execute(
        FunctionCall(
            "3",
            "finish_drift",
            {
                "skill_used": "alpha",
                "one_line": "完成一次动作",
                "next": "继续观察",
                "message_result": "sent",
            },
        )
    )
    assert finished.ok and state.finished
    assert sent == ["你好"]


@pytest.mark.asyncio
async def test_finish_drift_rejects_sent_without_message(tmp_path: Path) -> None:
    state = DriftRunState(frozenset({"alpha"}))
    registry = build_drift_tool_registry(workspace=tmp_path, state=state)
    result = await registry.execute(
        FunctionCall(
            "1",
            "finish_drift",
            {
                "skill_used": "alpha",
                "one_line": "完成",
                "next": "下一步",
                "message_result": "sent",
            },
        )
    )
    assert "requires successful" in str(result.result)
    assert not state.finished


@pytest.mark.asyncio
async def test_mount_server_copies_only_connected_mcp_tools(tmp_path: Path) -> None:
    async def remote_tool(value: str) -> str:
        return value

    shared = ToolRegistry()
    shared.register(
        Tool(
            "mcp_feed__lookup",
            "lookup",
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            remote_tool,
            source="mcp:feed/lookup",
        ),
        source_type="mcp",
        source_name="feed",
    )
    state = DriftRunState(frozenset({"alpha"}))
    registry = build_drift_tool_registry(
        workspace=tmp_path,
        state=state,
        shared_tools=shared,
        connected_servers=frozenset({"feed"}),
    )

    mounted = await registry.execute(FunctionCall("1", "mount_server", {"server": "feed"}))
    assert mounted.ok
    assert registry.has_tool("mcp_feed__lookup")
    unavailable = await registry.execute(
        FunctionCall("2", "mount_server", {"server": "missing"})
    )
    assert "unavailable" in str(unavailable.result)


@pytest.mark.asyncio
async def test_drift_reuses_shared_public_tools(tmp_path: Path) -> None:
    async def recall(**_: object) -> str:
        return "ok"

    shared = ToolRegistry()
    shared.register(
        Tool(
            "recall_memory",
            "recall",
            {"type": "object", "properties": {}},
            recall,
        )
    )
    registry = build_drift_tool_registry(
        workspace=tmp_path,
        state=DriftRunState(frozenset()),
        shared_tools=shared,
    )
    assert registry.has_tool("recall_memory")
