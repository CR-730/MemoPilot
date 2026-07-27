"""Channel 边界上的小型协议。

消息总线本体位于 :mod:`memopilot.bus`；这里仅保留 Channel 使用的回执和中断协议，
并继续导出消息类型以兼容现有适配器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memopilot.bus.events import InboundMessage, OutboundMessage
from memopilot.bus.queue import MessageBus


@dataclass(frozen=True, slots=True)
class SendReceipt:
    message_id: str


@dataclass(frozen=True, slots=True)
class InterruptAcknowledgement:
    message: str
    provider_uuid: str


class InterruptController(Protocol):
    async def request_interrupt(self, message: InboundMessage) -> InterruptAcknowledgement: ...


__all__ = [
    "InboundMessage",
    "InterruptAcknowledgement",
    "InterruptController",
    "MessageBus",
    "OutboundMessage",
    "SendReceipt",
]
