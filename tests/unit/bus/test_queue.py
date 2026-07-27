from __future__ import annotations

import asyncio

from memopilot.bus.events import InboundMessage, OutboundMessage
from memopilot.bus.queue import MessageBus
from memopilot.runtime.outbound import BusOutboundPort, OutboundDispatch


async def test_message_bus_keeps_prototype_inbound_and_outbound_contract() -> None:
    bus = MessageBus()
    inbound = InboundMessage("feishu", "user", "chat-1", "你好")

    await bus.publish_inbound(inbound)

    assert await bus.consume_inbound() is inbound

    received: list[OutboundMessage] = []

    async def receive(message: OutboundMessage) -> None:
        received.append(message)

    bus.subscribe_outbound("feishu", receive)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    try:
        await BusOutboundPort(bus).dispatch(
            OutboundDispatch(
                channel="feishu",
                chat_id="chat-1",
                content="嗯。",
                thinking="先接住",
            )
        )
        await asyncio.sleep(0)
    finally:
        bus.stop()
        await dispatch

    assert received == [
        OutboundMessage(
            channel="feishu",
            chat_id="chat-1",
            content="嗯。",
            thinking="先接住",
        )
    ]


def test_inbound_message_keeps_prototype_context_overrides() -> None:
    message = InboundMessage(
        "internal",
        "system",
        "job-1",
        "完成",
        metadata={
            "session_key_override": "feishu:chat-1",
            "context_channel": "feishu",
            "context_chat_id": "chat-1",
        },
    )

    assert message.session_key == "feishu:chat-1"
    assert message.context_channel == "feishu"
    assert message.context_chat_id == "chat-1"
