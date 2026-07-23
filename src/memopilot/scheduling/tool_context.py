"""定时工具的可信 Turn 上下文。"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ScheduleToolContext:
    session_key: str
    received_at: datetime
    default_timezone: str = "Asia/Shanghai"
    assert_current: Callable[[], None] | None = None


_CURRENT: ContextVar[ScheduleToolContext | None] = ContextVar(
    "memopilot_schedule_tool_context", default=None
)


def bind_schedule_tool_context(
    session_key: str,
    *,
    received_at: datetime,
    default_timezone: str = "Asia/Shanghai",
    assert_current: Callable[[], None] | None = None,
) -> Token[ScheduleToolContext | None]:
    return _CURRENT.set(
        ScheduleToolContext(
            session_key,
            received_at,
            default_timezone,
            assert_current,
        )
    )


def current_schedule_tool_context() -> ScheduleToolContext | None:
    return _CURRENT.get()


def reset_schedule_tool_context(token: Token[ScheduleToolContext | None]) -> None:
    _CURRENT.reset(token)


__all__ = [
    "ScheduleToolContext",
    "bind_schedule_tool_context",
    "current_schedule_tool_context",
    "reset_schedule_tool_context",
]
