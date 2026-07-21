"""为记忆工具提供当前 Turn 的会话作用域，不把内部字段暴露给模型。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryToolContext:
    session_key: str
    channel: str
    chat_id: str
    source_ref: str
    assert_current: Callable[[], None] | None = None
    fenced_write: Callable[[], AbstractContextManager[None]] | None = None


_CURRENT: ContextVar[MemoryToolContext | None] = ContextVar(
    "memopilot_memory_tool_context", default=None
)


def bind_memory_tool_context(
    session_key: str,
    *,
    source_ref: str = "",
    assert_current: Callable[[], None] | None = None,
    fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
) -> Token[MemoryToolContext | None]:
    channel, separator, chat_id = session_key.partition(":")
    return _CURRENT.set(
        MemoryToolContext(
            session_key=session_key,
            channel=channel if separator else "",
            chat_id=chat_id if separator else "",
            source_ref=source_ref,
            assert_current=assert_current,
            fenced_write=fenced_write,
        )
    )


def current_memory_tool_context() -> MemoryToolContext | None:
    return _CURRENT.get()


def reset_memory_tool_context(token: Token[MemoryToolContext | None]) -> None:
    _CURRENT.reset(token)


__all__ = [
    "MemoryToolContext",
    "bind_memory_tool_context",
    "current_memory_tool_context",
    "reset_memory_tool_context",
]
