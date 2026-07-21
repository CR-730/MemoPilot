"""原型兼容的单一事件总线：有序 Gate、隔离 Observer 与后台派发。"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast, overload

logger = logging.getLogger(__name__)

E = TypeVar("E")
type EventKey = str | type[object]
EventCallback = Callable[
    [dict[str, object]],
    Awaitable[Mapping[str, Any] | None] | Mapping[str, Any] | None,
]
type TypedCallback = Callable[[Any], Awaitable[Any | None] | Any | None]


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


@dataclass(frozen=True, slots=True)
class _RegisteredHandler:
    handler_id: str
    key: EventKey
    callback: TypedCallback
    observer: bool
    priority: int
    sequence: int


class EventSubscription[E]:
    """可撤销订阅；显式持有令牌的调用方可精确清理。"""

    def __init__(self, bus: EventBus, sequence: int) -> None:
        self._bus = bus
        self._sequence = sequence
        self._active = True

    def unsubscribe(self) -> None:
        if self._active:
            self._bus._unsubscribe(self._sequence)
            self._active = False


class EventBus:
    """同一实例同时承载 Gate 和 Observer，不维护第二套总线。"""

    def __init__(self, handlers: tuple[EventHandler, ...] = ()) -> None:
        self._handlers: list[_RegisteredHandler] = []
        self._next_sequence = 0
        self._observe_queue: asyncio.Queue[tuple[object, object | None]] | None = None
        self._observe_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self.diagnostics: list[EventDiagnostic] = []
        self.register_many(handlers)

    def register_many(
        self,
        handlers: tuple[EventHandler, ...],
    ) -> tuple[EventSubscription[object], ...]:
        identities = {(item.key, item.handler_id) for item in self._handlers}
        for handler in handlers:
            identity = (handler.event, handler.handler_id)
            if identity in identities:
                raise ValueError(f"EventHandler 重复: {handler.event}/{handler.handler_id}")
            identities.add(identity)
        return tuple(
            self.on(
                handler.event,
                handler.callback,
                observer=handler.observer,
                priority=handler.priority,
                handler_id=handler.handler_id,
            )
            for handler in handlers
        )

    def on(
        self,
        event: str | type[E],
        handler: Callable[[E], Awaitable[E | None] | E | None] | EventCallback,
        *,
        observer: bool = False,
        priority: int = 100,
        handler_id: str | None = None,
    ) -> EventSubscription[E]:
        if self._closed:
            raise RuntimeError("EventBus 已关闭")
        resolved_id = handler_id or _handler_name(handler)
        if any(item.key == event and item.handler_id == resolved_id for item in self._handlers):
            raise ValueError(f"EventHandler 重复: {_event_name(event)}/{resolved_id}")
        sequence = self._next_sequence
        self._next_sequence += 1
        self._handlers.append(
            _RegisteredHandler(
                resolved_id,
                cast(EventKey, event),
                cast(TypedCallback, handler),
                observer,
                priority,
                sequence,
            )
        )
        self._handlers.sort(key=lambda item: (item.priority, item.sequence))
        return EventSubscription(self, sequence)

    @overload
    async def emit(self, event: str, payload: Mapping[str, object]) -> dict[str, object]: ...

    @overload
    async def emit(self, event: E, payload: None = None) -> E: ...

    async def emit(
        self,
        event: E | str,
        payload: Mapping[str, object] | None = None,
    ) -> E | dict[str, object]:
        current: Any = dict(payload) if payload is not None else event
        for handler in self._matching(event, observer=False):
            result = handler.callback(current)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                current = result
        return cast(E | dict[str, object], current)

    async def observe(self, event: object, payload: Mapping[str, object] | None = None) -> None:
        current = dict(payload) if payload is not None else event
        for handler in self._matching(event, observer=True):
            await self._run_observer(event, current, handler)

    async def fanout(self, event: object, payload: Mapping[str, object] | None = None) -> None:
        current = dict(payload) if payload is not None else event
        handlers = self._matching(event, observer=True)
        if not handlers:
            return
        results = await asyncio.gather(
            *(self._run_observer(event, current, handler) for handler in handlers)
        )
        failed_count = results.count(False)
        if failed_count:
            logger.warning(
                "fanout completed with observer errors: event=%s failed=%d total=%d",
                _event_name(event),
                failed_count,
                len(handlers),
            )

    def enqueue(self, event: object, payload: Mapping[str, object] | None = None) -> None:
        if self._closing or self._closed:
            raise RuntimeError("EventBus 正在关闭或已关闭")
        queue = self._ensure_observe_queue()
        queue.put_nowait((event, dict(payload) if payload is not None else None))

    async def drain(self) -> None:
        queue = self._observe_queue
        if queue is None:
            return
        self._ensure_observe_task()
        await queue.join()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            await self.drain()
            self._closed = True
            task = self._observe_task
            if task is None:
                return
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _matching(self, event: object, *, observer: bool) -> list[_RegisteredHandler]:
        key: EventKey = event if isinstance(event, str) else type(event)
        return [
            handler
            for handler in self._handlers
            if handler.key == key and handler.observer is observer
        ]

    async def _run_observer(
        self,
        event: object,
        payload: object,
        handler: _RegisteredHandler,
    ) -> bool:
        try:
            result = handler.callback(payload)
            if inspect.isawaitable(result):
                await result
            return True
        except Exception as exc:
            self.diagnostics.append(
                EventDiagnostic(handler.handler_id, _event_name(event), type(exc).__name__)
            )
            logger.exception(
                "observer error for %s handler=%s",
                _event_name(event),
                handler.handler_id,
            )
            return False

    def _unsubscribe(self, sequence: int) -> None:
        self._handlers = [item for item in self._handlers if item.sequence != sequence]

    def _ensure_observe_queue(self) -> asyncio.Queue[tuple[object, object | None]]:
        if self._observe_queue is None:
            self._observe_queue = asyncio.Queue()
        self._ensure_observe_task()
        return self._observe_queue

    def _ensure_observe_task(self) -> None:
        if self._closed:
            return
        if self._observe_task is not None and not self._observe_task.done():
            return
        task = asyncio.create_task(self._run_observe_queue(), name="event_bus_observe_queue")
        self._observe_task = task
        task.add_done_callback(self._on_observe_task_done)

    async def _run_observe_queue(self) -> None:
        while True:
            queue = self._observe_queue
            if queue is None:
                return
            event, payload = await queue.get()
            try:
                await self.fanout(
                    event,
                    cast(Mapping[str, object] | None, payload),
                )
            finally:
                queue.task_done()

    def _on_observe_task_done(self, task: asyncio.Task[None]) -> None:
        if self._observe_task is task:
            self._observe_task = None
        if self._closed or task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "event dispatcher stopped unexpectedly",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        if self._observe_queue is not None:
            self._ensure_observe_task()


def _handler_name(handler: object) -> str:
    return str(getattr(handler, "__qualname__", getattr(handler, "__name__", repr(handler))))


def _event_name(event: object) -> str:
    return event if isinstance(event, str) else type(event).__name__


__all__ = [
    "EventBus",
    "EventCallback",
    "EventDiagnostic",
    "EventHandler",
    "EventSubscription",
]
