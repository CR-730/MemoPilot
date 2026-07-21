"""统一工具调用前后 Hook 合同。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

HookEvent = Literal["pre_tool_use", "post_tool_use", "post_tool_error"]
HookDecision = Literal["pass", "deny"]
ToolExecStatus = Literal["success", "denied", "error"]


@dataclass(slots=True)
class ToolExecutionRequest:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    source: str = "passive"
    session_key: str = ""
    channel: str = ""
    chat_id: str = ""
    request_text: str = ""
    tool_batch: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tool_batch_index: int = 0


@dataclass(slots=True)
class HookContext:
    event: HookEvent
    request: ToolExecutionRequest
    current_arguments: dict[str, Any]
    result: Any = ""
    error: str = ""


@dataclass(slots=True)
class HookOutcome:
    decision: HookDecision = "pass"
    updated_input: dict[str, Any] | None = None
    extra_message: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class HookTraceItem:
    """一次 Hook 匹配/执行的可审计记录，与固定原型合同一致。"""

    hook_name: str
    event: HookEvent
    matched: bool
    decision: HookDecision = "pass"
    reason: str = ""
    extra_message: str = ""


@dataclass(frozen=True, slots=True)
class ToolHookDecision:
    """兼容阶段 5 早期的函数式 pre-hook 返回值。"""

    arguments: Mapping[str, Any] | None = None
    denied: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.denied and not (self.reason or "").strip():
            raise ValueError("Hook 拒绝工具调用时必须提供 reason")


BeforeToolHook = Callable[
    [str, dict[str, object]],
    Awaitable[ToolHookDecision | None],
]
AfterToolHook = Callable[[str, object], Awaitable[None]]


class ToolHook:
    """原型 matches/run Hook，并兼容已有 before/after 函数式用法。"""

    def __init__(
        self,
        hook_id: str,
        before: BeforeToolHook | None = None,
        after: AfterToolHook | None = None,
        *,
        event: HookEvent | None = None,
    ) -> None:
        if not hook_id.strip():
            raise ValueError("ToolHook hook_id 不能为空")
        if event is None and before is None and after is None:
            raise ValueError(f"ToolHook {hook_id} 至少需要 before、after 或 event")
        self.hook_id = hook_id
        self.name = hook_id
        self.before = before
        self.after = after
        self.event = event

    def matches(self, context: HookContext) -> bool:
        del context
        return True

    async def run(self, context: HookContext) -> HookOutcome:
        del context
        return HookOutcome()


__all__ = [
    "AfterToolHook",
    "BeforeToolHook",
    "HookContext",
    "HookEvent",
    "HookOutcome",
    "HookTraceItem",
    "ToolExecStatus",
    "ToolExecutionRequest",
    "ToolHook",
    "ToolHookDecision",
]
