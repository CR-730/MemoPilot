"""有序 Gate 与隔离型 Observer 事件总线。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

EventCallback = Callable[
    [dict[str, object]],
    Awaitable[Mapping[str, Any] | None],
]


@dataclass(frozen=True, slots=True)
class EventHandler:
    handler_id: str
    event: str
    callback: EventCallback
    observer: bool = False
    priority: int = 100


@dataclass(frozen=True, slots=True)
class EventDiagnostic:
    handler_id: str
    event: str
    error_type: str


class EventBus:
    def __init__(self, handlers: tuple[EventHandler, ...] = ()) -> None:
        identities: set[tuple[str, str]] = set()
        for handler in handlers:
            identity = (handler.event, handler.handler_id)
            if identity in identities:
                raise ValueError(f"EventHandler 重复: {handler.event}/{handler.handler_id}")
            identities.add(identity)
        self._handlers = tuple(
            sorted(handlers, key=lambda item: (item.event, item.priority, item.handler_id))
        )
        self.diagnostics: list[EventDiagnostic] = []

    async def emit(self, event: str, payload: Mapping[str, object]) -> dict[str, object]:
        current = dict(payload)
        for handler in self._handlers:
            if handler.event != event or handler.observer:
                continue
            updated = await handler.callback(dict(current))
            if updated is not None:
                current = dict(updated)
        for handler in self._handlers:
            if handler.event != event or not handler.observer:
                continue
            try:
                await handler.callback(dict(current))
            except Exception as exc:
                self.diagnostics.append(
                    EventDiagnostic(handler.handler_id, event, type(exc).__name__)
                )
        return current


__all__ = ["EventBus", "EventCallback", "EventDiagnostic", "EventHandler"]

