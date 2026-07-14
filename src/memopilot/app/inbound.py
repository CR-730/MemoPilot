"""飞书 Channel 到 operational inbox/outbox 的适配。"""

from __future__ import annotations

import asyncio
from uuid import NAMESPACE_URL, uuid5

from memopilot.channels.contracts import (
    InboundMessage,
    InterruptAcknowledgement,
    MessageBus,
)
from memopilot.tasks.operational import (
    EnqueueResult,
    InboundCommand,
    InterruptCommand,
    OperationalRepository,
)


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
    def __init__(self, repository: OperationalRepository) -> None:
        self._repository = repository

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
        await asyncio.to_thread(self._repository.request_interrupt, command)
        provider_uuid = str(uuid5(NAMESPACE_URL, f"feishu:interrupt:{stable_event_id}"))
        return InterruptAcknowledgement(message="已请求中断。", provider_uuid=provider_uuid)


__all__ = ["InboundBridge", "OperationalInterruptController"]
