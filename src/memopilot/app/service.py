"""App 进程：启动 Channel，并把 AgentLoop 的结果发送到外部渠道。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from memopilot.bus.events import OutboundMessage
from memopilot.channels.contracts import MessageBus


class ChannelLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class OutboundTransport(Protocol):
    async def send(
        self,
        chat_id: str,
        message: str,
        *,
        provider_uuid: str,
        metadata: dict[str, object],
    ) -> object: ...


class GatewayService:
    """保持原型的轻量边界：Channel 只进 Bus，AgentLoop 负责处理消息。"""

    def __init__(
        self,
        *,
        channel: ChannelLifecycle,
        bus: MessageBus,
        outbound_transports: dict[str, object] | None = None,
    ) -> None:
        self._channel = channel
        self._bus = bus
        self._outbound_transports = outbound_transports or {}
        self._tasks: tuple[asyncio.Task[None], ...] = ()

    async def start(self) -> None:
        if self._tasks:
            return
        for channel_name, transport in self._outbound_transports.items():
            self._bus.subscribe_outbound(channel_name, self._send_outbound(transport))
        await self._channel.start()
        if self._outbound_transports:
            self._tasks = (
                asyncio.create_task(
                    self._bus.dispatch_outbound(),
                    name="memopilot-bus-outbound",
                ),
            )

    async def stop(self) -> None:
        self._bus.stop()
        tasks, self._tasks = self._tasks, ()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._channel.stop()

    @staticmethod
    def _send_outbound(transport: object) -> Callable[[OutboundMessage], Awaitable[None]]:
        async def send(message: OutboundMessage) -> None:
            sender = cast(OutboundTransport, transport).send
            await sender(
                message.chat_id,
                message.content,
                provider_uuid=str(message.metadata.get("provider_uuid") or ""),
                metadata=message.metadata,
            )

        return send


AppService = GatewayService

__all__ = ["GatewayService", "ChannelLifecycle", "AppService"]
