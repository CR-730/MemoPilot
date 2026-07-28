"""Channel 边界上的小型协议。

消息总线本体位于 :mod:`memopilot.bus`；这里仅保留 Channel 使用的回执和中断协议，
并继续导出消息类型以兼容现有适配器。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from memopilot.bus.events import InboundMessage


@dataclass(frozen=True, slots=True)
class SendReceipt:
    message_id: str


@dataclass(frozen=True, slots=True)
class InterruptAcknowledgement:
    message: str
    provider_uuid: str


InboundHandler = Callable[[InboundMessage], Awaitable[object]]


__all__ = [
    "InboundMessage",
    "InterruptAcknowledgement",
    "InboundHandler",
    "SendReceipt",
]
