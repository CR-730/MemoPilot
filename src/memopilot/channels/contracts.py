"""Channel 与 App/Delivery 之间的稳定数据合同。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class InboundMessage:
    channel: str
    sender: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    media: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        return f"{self.channel}:{self.chat_id}"


@dataclass(frozen=True, slots=True)
class SendReceipt:
    message_id: str


@dataclass(frozen=True, slots=True)
class InterruptAcknowledgement:
    message: str
    provider_uuid: str


class InterruptController(Protocol):
    async def request_interrupt(self, message: InboundMessage) -> InterruptAcknowledgement: ...


class MessageBus:
    """Channel 到 InboundBridge 的进程内异步边界。"""

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._subscribers: list[Callable[[InboundMessage], Awaitable[None]]] = []

    async def publish_inbound(self, message: InboundMessage) -> None:
        subscribers = tuple(self._subscribers)
        if not subscribers:
            await self._inbound.put(message)
            return
        for subscriber in subscribers:
            await subscriber(message)

    async def consume_inbound(self) -> InboundMessage:
        return await self._inbound.get()

    def subscribe_inbound(
        self,
        callback: Callable[[InboundMessage], Awaitable[None]],
    ) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    @property
    def inbound_size(self) -> int:
        return self._inbound.qsize()


__all__ = [
    "InboundMessage",
    "InterruptAcknowledgement",
    "InterruptController",
    "MessageBus",
    "SendReceipt",
]
