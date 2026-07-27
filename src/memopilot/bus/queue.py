import asyncio
import logging
from collections.abc import Awaitable, Callable

from memopilot.bus.events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)


class MessageBus:
    """Agent 与各 Channel 之间的异步消息总线。"""

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._subscribers: dict[str, list[Callable[[OutboundMessage], Awaitable[None]]]] = {}
        self._running = False
        self._inbound_subscribers: list[Callable[[InboundMessage], Awaitable[None]]] = []

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Channel → Agent。"""
        await self._inbound.put(msg)
        subscribers = tuple(self._inbound_subscribers)
        for callback in subscribers:
            await callback(msg)
        if subscribers:
            # 旧版订阅者直接处理消息，不再让兼容队列残留一份副本。
            try:
                self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                pass

    async def consume_inbound(self) -> InboundMessage:
        """阻塞直到有消息可消费。"""
        return await self._inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Agent → Channel。"""
        await self._outbound.put(msg)

    def subscribe_outbound(
        self,
        channel: str,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """订阅指定 Channel 的出站消息。"""
        self._subscribers.setdefault(channel, []).append(callback)

    def subscribe_inbound(
        self,
        callback: Callable[[InboundMessage], Awaitable[None]],
    ) -> Callable[[], None]:
        self._inbound_subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._inbound_subscribers:
                self._inbound_subscribers.remove(callback)

        return unsubscribe

    async def dispatch_outbound(self) -> None:
        """将出站消息分发给对应 Channel；失败后按原型重试一次。"""
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
                for callback in self._subscribers.get(msg.channel, []):
                    try:
                        await callback(msg)
                    except Exception as first_error:
                        logger.warning(
                            "分发消息到 %s 首次失败，2s 后重试: %s",
                            msg.channel,
                            first_error,
                        )
                        await asyncio.sleep(2)
                        try:
                            await callback(msg)
                        except Exception as second_error:
                            logger.error(
                                "分发消息到 %s 重试仍失败，发送降级通知: %s",
                                msg.channel,
                                second_error,
                            )
                            fallback = OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="（消息发送失败，请稍后重试）",
                            )
                            try:
                                await callback(fallback)
                            except Exception:
                                logger.error(
                                    "降级通知也失败，消息彻底丢失 channel=%s chat_id=%s",
                                    msg.channel,
                                    msg.chat_id,
                                )
            except TimeoutError:
                continue

    def stop(self) -> None:
        self._running = False

    @property
    def inbound_size(self) -> int:
        return self._inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self._outbound.qsize()


__all__ = ["MessageBus"]
