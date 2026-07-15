"""飞书 Channel 到 operational inbox/outbox 的适配。"""

from __future__ import annotations

import asyncio
import logging
from uuid import NAMESPACE_URL, uuid5

from memopilot.channels.contracts import (
    InboundMessage,
    InterruptAcknowledgement,
    MessageBus,
)
from memopilot.tasks.interrupts import InterruptSignalPort
from memopilot.tasks.operational import (
    EnqueueResult,
    InboundCommand,
    InterruptCommand,
    OperationalRepository,
)

logger = logging.getLogger(__name__)


class InboundBridge:
    def __init__(self, repository: OperationalRepository) -> None:
        self._repository = repository

    async def handle(self, message: InboundMessage) -> EnqueueResult:
        event_id = str(message.metadata.get("event_id") or "").strip()
        message_id = str(message.metadata.get("message_id") or "").strip()
        if not message_id:
            raise ValueError("飞书入站消息缺少 message_id")
        command = InboundCommand(
            event_id=event_id or message_id,
            message_id=message_id,
            session_key=message.session_key,
            channel=message.channel,
            chat_id=message.chat_id,
            payload={
                "channel": message.channel,
                "sender": message.sender,
                "chat_id": message.chat_id,
                "text": message.content,
                "media": list(message.media),
                "metadata": dict(message.metadata),
            },
            received_at=message.timestamp,
        )
        return await asyncio.to_thread(self._repository.accept_inbound, command)

    async def run_once(self, bus: MessageBus) -> EnqueueResult:
        return await self.handle(await bus.consume_inbound())


class OperationalInterruptController:
    def __init__(
        self,
        repository: OperationalRepository,
        *,
        signal: InterruptSignalPort | None = None,
    ) -> None:
        self._repository = repository
        self._signal = signal

    async def request_interrupt(self, message: InboundMessage) -> InterruptAcknowledgement:
        event_id = str(message.metadata.get("event_id") or "").strip()
        message_id = str(message.metadata.get("message_id") or "").strip()
        if not message_id:
            raise ValueError("飞书中断消息缺少 message_id")
        stable_event_id = event_id or message_id
        command = InterruptCommand(
            event_id=stable_event_id,
            message_id=message_id,
            session_key=message.session_key,
            channel=message.channel,
            chat_id=message.chat_id,
            requested_at=message.timestamp,
        )
        result = await asyncio.to_thread(self._repository.request_interrupt, command)
        if result.target_run_id is not None and self._signal is not None:
            try:
                await self._signal.publish(result.target_run_id)
            except Exception as exc:
                logger.warning("Redis 中断通知写入失败，Worker 将回退轮询 SQLite: %s", exc)
        provider_uuid = str(uuid5(NAMESPACE_URL, f"feishu:interrupt:{stable_event_id}"))
        message_text = (
            "本轮已中断。你可以继续补充要求，我会接着这件事处理。"
            if result.target_run_id is not None
            else "当前没有正在执行的任务。"
        )
        return InterruptAcknowledgement(message=message_text, provider_uuid=provider_uuid)


__all__ = ["InboundBridge", "OperationalInterruptController"]
