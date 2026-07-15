"""统一工具调用前后 Hook 合同。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolHookDecision:
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


@dataclass(frozen=True, slots=True)
class ToolHook:
    hook_id: str
    before: BeforeToolHook | None = None
    after: AfterToolHook | None = None

    def __post_init__(self) -> None:
        if not self.hook_id.strip():
            raise ValueError("ToolHook hook_id 不能为空")
        if self.before is None and self.after is None:
            raise ValueError(f"ToolHook {self.hook_id} 至少需要 before 或 after")


__all__ = ["AfterToolHook", "BeforeToolHook", "ToolHook", "ToolHookDecision"]

