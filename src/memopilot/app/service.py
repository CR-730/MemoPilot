"""App 进程的飞书入站与 Outbox 派发循环。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from memopilot.app.inbound import InboundBridge
from memopilot.channels.contracts import InboundMessage, MessageBus
from memopilot.tasks.outbox import OutboxDispatcher
from memopilot.tasks.redis_queue import RedisTaskQueue

logger = logging.getLogger(__name__)


class ChannelLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class AppService:
    """只负责编排 App 所属组件，不承载 Agent Runtime。"""

    def __init__(
        self,
        *,
        channel: ChannelLifecycle,
        bus: MessageBus,
        bridge: InboundBridge,
        queue: RedisTaskQueue,
        outbox: OutboxDispatcher,
        clock: Callable[[], datetime] | None = None,
        idle_interval: float = 0.05,
    ) -> None:
        if idle_interval <= 0:
            raise ValueError("idle_interval 必须大于 0")
        self._channel = channel
        self._bus = bus
        self._bridge = bridge
        self._queue = queue
        self._outbox = outbox
        self._clock = clock or (lambda: datetime.now(UTC))
        self._idle_interval = idle_interval
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._unsubscribe_inbound: Callable[[], None] | None = None

    async def start(self) -> None:
        if self._tasks or self._unsubscribe_inbound is not None:
            return
        await self._queue.ensure_consumer_groups()
        self._unsubscribe_inbound = self._bus.subscribe_inbound(self._persist_inbound)
        try:
            await self._channel.start()
        except BaseException:
            self._unsubscribe_inbound()
            self._unsubscribe_inbound = None
            raise
        self._tasks = (
            asyncio.create_task(self._outbox_loop(), name="memopilot-app-outbox"),
        )

    async def stop(self) -> None:
        tasks, self._tasks = self._tasks, ()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await self._channel.stop()
        finally:
            if self._unsubscribe_inbound is not None:
                self._unsubscribe_inbound()
                self._unsubscribe_inbound = None

    async def _persist_inbound(self, message: InboundMessage) -> None:
        while True:
            try:
                await self._bridge.handle(message)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("飞书入站消息持久化失败，将保留消息并重试")
                await asyncio.sleep(self._idle_interval)

    async def _outbox_loop(self) -> None:
        while True:
            try:
                dispatched = await self._outbox.dispatch_one(now=self._clock())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Outbox 发布失败，将按持久化重试状态继续派发")
                dispatched = False
            if not dispatched:
                await asyncio.sleep(self._idle_interval)


__all__ = ["AppService", "ChannelLifecycle"]
