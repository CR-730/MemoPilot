from __future__ import annotations

from typing import cast

from memopilot.extensions.hooks import (
    HookContext,
    HookOutcome,
    HookTraceItem,
    ToolExecutionRequest,
    ToolHook,
    ToolHookDecision,
)
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolObservation, ToolRegistry


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
    assert result.status == "success"
    assert result.hook_trace == (
        HookTraceItem(
            hook_name="rewrite",
            event="pre_tool_use",
            matched=True,
        ),
    )


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
    assert result.status == "denied"
    assert result.error_type == "hook_denied"
    assert result.error_message == "命令不安全"
    assert result.hook_trace == (
        HookTraceItem(
            hook_name="safety",
            event="pre_tool_use",
            matched=True,
            decision="deny",
            reason="命令不安全",
        ),
    )


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
    assert result.status == "error"
    assert result.hook_trace == (
        HookTraceItem(
            hook_name="broken",
            event="pre_tool_use",
            matched=True,
            reason="hook failed: secret details",
        ),
    )


async def test_after_hook_observes_failed_tool_result() -> None:
    observed: list[bool] = []

    async def handler() -> str:
        raise RuntimeError("failed")

    async def after(tool_name: str, observation: object) -> None:
        del tool_name
        observed.append(cast(ToolObservation, observation).ok)

    registry = ToolRegistry(
        (Tool("demo", "demo", {"type": "object"}, handler),),
        hooks=(ToolHook("audit", after=after),),
    )

    result = await registry.execute(FunctionCall("c1", "demo", {}))

    assert result.ok is False
    assert observed == [False]
    assert result.hook_trace == (
        HookTraceItem(
            hook_name="audit",
            event="post_tool_error",
            matched=True,
        ),
    )


async def test_rich_hook_context_matches_and_preserves_extra_messages() -> None:
    seen: list[HookContext] = []

    class ContextHook(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return context.request.tool_name == "demo"

        async def run(self, context: HookContext) -> HookOutcome:
            seen.append(context)
            return HookOutcome(
                updated_input={"value": "rewritten"},
                extra_message="已安全改写",
            )

    async def handler(value: str) -> str:
        return value

    hook = ContextHook("context", event="pre_tool_use")
    registry = ToolRegistry(
        [
            Tool(
                "demo",
                "demo",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                handler,
            )
        ],
        hooks=[hook],
    )

    result = await registry.execute(
        FunctionCall("c1", "demo", {"value": "original"}),
        request=ToolExecutionRequest(
            call_id="c1",
            tool_name="demo",
            arguments={"value": "original"},
            source="passive",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            request_text="执行 demo",
        ),
    )

    assert result.ok is True
    assert result.result == "rewritten"
    assert result.extra_messages == ("已安全改写",)
    assert result.hook_trace == (
        HookTraceItem(
            hook_name="context",
            event="pre_tool_use",
            matched=True,
            extra_message="已安全改写",
        ),
    )
    assert seen[0].request.session_key == "feishu:chat-1"
    assert seen[0].request.source == "passive"


async def test_post_success_hook_failure_is_fail_open_but_traced() -> None:
    class BrokenPostHook(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return True

        async def run(self, context: HookContext) -> HookOutcome:
            raise RuntimeError("audit unavailable")

    async def handler() -> str:
        return "ok"

    result = await ToolRegistry(
        [Tool("demo", "demo", {"type": "object"}, handler)],
        hooks=[BrokenPostHook("audit", event="post_tool_use")],
    ).execute(FunctionCall("c1", "demo", {}))

    assert result.ok is True
    assert result.result == "ok"
    assert result.hook_trace[-1] == HookTraceItem(
        hook_name="audit",
        event="post_tool_use",
        matched=True,
        reason="hook failed: audit unavailable",
    )


async def test_rich_hook_trace_keeps_unmatched_pre_and_post_items() -> None:
    class NeverMatches(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return False

    async def handler() -> str:
        return "ok"

    result = await ToolRegistry(
        [Tool("demo", "demo", {"type": "object"}, handler)],
        hooks=[
            NeverMatches("pre", event="pre_tool_use"),
            NeverMatches("post", event="post_tool_use"),
        ],
    ).execute(FunctionCall("c1", "demo", {}))

    assert result.hook_trace == (
        HookTraceItem("pre", "pre_tool_use", matched=False),
        HookTraceItem("post", "post_tool_use", matched=False),
    )
