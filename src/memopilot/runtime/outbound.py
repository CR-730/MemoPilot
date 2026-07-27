from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol

from memopilot.bus.events import OutboundMessage
from memopilot.runtime.common_tools.message_push import MessagePushTool


@dataclass
class OutboundDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    media: list[str] = field(default_factory=list)


class OutboundPort(Protocol):
    async def dispatch(self, outbound: OutboundDispatch) -> bool: ...


class BusOutboundPort:
    def __init__(self, bus: Any) -> None:
        self._bus = bus

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        maybe = self._bus.publish_outbound(
            OutboundMessage(
                channel=outbound.channel,
                chat_id=outbound.chat_id,
                content=outbound.content,
                thinking=outbound.thinking,
                metadata=dict(outbound.metadata or {}),
                media=list(outbound.media or []),
            )
        )
        if inspect.isawaitable(maybe):
            await maybe
        return True


class PushToolOutboundPort:
    """通过公共 message_push 工具发送后台消息。"""

    def __init__(self, message_push: MessagePushTool) -> None:
        self._message_push = message_push

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        result = await self._message_push.execute(
            channel=outbound.channel,
            chat_id=outbound.chat_id,
            message=outbound.content,
        )
        return "已发送" in result


__all__ = [
    "BusOutboundPort",
    "OutboundDispatch",
    "OutboundPort",
    "PushToolOutboundPort",
]
