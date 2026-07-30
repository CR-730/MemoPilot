from __future__ import annotations

from memopilot.extensions.hooks import HookContext, HookOutcome, ToolHook
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolExecutor, ToolRegistry


def _executor(handler, hooks=()):
    async def tool(command: str = ""):
        return await handler(command)

    registry = ToolRegistry(
        [
            Tool(
                "shell", "shell",
                {"type": "object", "properties": {"command": {"type": "string"}}},
                tool,
            )
        ]
    )
    return ToolExecutor(registry, hooks)


async def test_pre_hook_rewrites_arguments() -> None:
    class Rewrite(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return True

        async def run(self, context: HookContext) -> HookOutcome:
            return HookOutcome(updated_input={"command": "safe"})

    seen: list[str] = []

    async def handler(command: str) -> str:
        seen.append(command)
        return "ok"

    result = await _executor(
        handler, [Rewrite("rewrite", event="pre_tool_use")]
    ).execute(FunctionCall("1", "shell", {"command": "danger"}))
    assert result.ok and seen == ["safe"]
    assert result.original_arguments == {"command": "danger"}
    assert result.final_arguments == {"command": "safe"}


async def test_pre_hook_denies_and_errors_closed() -> None:
    class Deny(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return True

        async def run(self, context: HookContext) -> HookOutcome:
            return HookOutcome(decision="deny", reason="blocked")

    async def handler(command: str) -> str:
        raise AssertionError("must not run")

    result = await _executor(
        handler, [Deny("deny", event="pre_tool_use")]
    ).execute(FunctionCall("1", "shell", {"command": "x"}))
    assert result.status == "denied" and result.error_type == "hook_denied"


async def test_post_hook_observes_error() -> None:
    observed: list[str] = []

    class Audit(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return True

        async def run(self, context: HookContext) -> HookOutcome:
            observed.append(context.event)
            return HookOutcome(extra_message="audited")

    async def handler(command: str) -> str:
        raise RuntimeError("boom")

    result = await _executor(
        handler, [Audit("audit", event="post_tool_error")]
    ).execute(FunctionCall("1", "shell", {"command": "x"}))
    assert not result.ok and observed == ["post_tool_error"]
    assert result.extra_messages == ("audited",)


async def test_derived_registry_keeps_registered_hooks() -> None:
    class DenyShell(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return context.request.tool_name == "shell"

        async def run(self, context: HookContext) -> HookOutcome:
            return HookOutcome(decision="deny", reason="background shell blocked")

    async def main_tool() -> str:
        return "main"

    async def shell(command: str) -> str:
        raise AssertionError(f"shell must not run: {command}")

    executor = ToolExecutor(
        ToolRegistry([Tool("main", "main", {"type": "object"}, main_tool)]),
        [DenyShell("shell-safety", event="pre_tool_use")],
    )
    drift_registry = ToolRegistry(
        [
            Tool(
                "shell",
                "shell",
                {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                shell,
            )
        ]
    )

    result = await executor.for_registry(drift_registry).execute(
        FunctionCall("1", "shell", {"command": "danger"})
    )

    assert result.status == "denied"
    assert result.error_message == "background shell blocked"
