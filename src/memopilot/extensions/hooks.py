"""统一工具调用前后 Hook 合同。"""

from __future__ import annotations

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


class ToolHook:
    """统一的工具 Hook 合同。"""

    def __init__(
        self,
        hook_id: str,
        *,
        event: HookEvent,
    ) -> None:
        if not hook_id.strip():
            raise ValueError("ToolHook hook_id 不能为空")
        self.hook_id = hook_id
        self.name = hook_id
        self.event = event

    def matches(self, context: HookContext) -> bool:
        del context
        return True

    async def run(self, context: HookContext) -> HookOutcome:
        del context
        return HookOutcome()


__all__ = [
    "HookContext",
    "HookEvent",
    "HookOutcome",
    "HookTraceItem",
    "ToolExecStatus",
    "ToolExecutionRequest",
    "ToolHook",
]
