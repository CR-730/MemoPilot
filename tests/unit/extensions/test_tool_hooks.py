from __future__ import annotations

from memopilot.extensions.hooks import ToolHook, ToolHookDecision
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolRegistry


async def test_tool_hook_rewrites_arguments_and_records_trace() -> None:
    received: list[str] = []

    async def handler(command: str) -> str:
        received.append(command)
        return "ok"

    async def rewrite(tool_name: str, arguments: dict[str, object]) -> ToolHookDecision:
        assert tool_name == "shell"
        return ToolHookDecision(arguments={"command": "safe"})

    registry = ToolRegistry(
        (
            Tool(
                "shell",
                "shell",
                {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                handler,
            ),
        ),
        hooks=(ToolHook("rewrite", before=rewrite),),
    )

    result = await registry.execute(FunctionCall("c1", "shell", {"command": "danger"}))

    assert result.ok is True
    assert received == ["safe"]
    assert result.original_arguments == {"command": "danger"}
    assert result.final_arguments == {"command": "safe"}
    assert result.hook_trace == ("rewrite:rewritten",)


async def test_tool_hook_can_deny_without_calling_tool() -> None:
    called = False

    async def handler(command: str) -> str:
        nonlocal called
        called = True
        return command

    async def deny(tool_name: str, arguments: dict[str, object]) -> ToolHookDecision:
        del tool_name, arguments
        return ToolHookDecision(denied=True, reason="命令不安全")

    registry = ToolRegistry(
        (Tool("shell", "shell", {"type": "object"}, handler),),
        hooks=(ToolHook("safety", before=deny),),
    )

    result = await registry.execute(FunctionCall("c1", "shell", {}))

    assert called is False
    assert result.ok is False
    assert result.error_type == "hook_denied"
    assert result.error_message == "命令不安全"
    assert result.hook_trace == ("safety:denied",)


async def test_before_hook_exception_fails_closed() -> None:
    async def handler() -> str:
        return "must not run"

    async def broken(tool_name: str, arguments: dict[str, object]) -> ToolHookDecision:
        del tool_name, arguments
        raise RuntimeError("secret details")

    registry = ToolRegistry(
        (Tool("demo", "demo", {"type": "object"}, handler),),
        hooks=(ToolHook("broken", before=broken),),
    )

    result = await registry.execute(FunctionCall("c1", "demo", {}))

    assert result.ok is False
    assert result.error_type == "hook_error"
    assert result.hook_trace == ("broken:error",)

