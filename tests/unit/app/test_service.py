from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from memopilot.app.service import AppService
from memopilot.channels.contracts import InboundMessage, MessageBus


class _Channel:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class _Queue:
    def __init__(self) -> None:
        self.ready = False

    async def ensure_consumer_groups(self) -> None:
        self.ready = True


class _Bridge:
    def __init__(self) -> None:
        self.messages: list[InboundMessage] = []

    async def handle(self, message: InboundMessage) -> object:
        self.messages.append(message)
        return object()


class _Outbox:
    def __init__(self) -> None:
        self.calls = 0

    async def dispatch_one(self, *, now: datetime) -> bool:
        self.calls += 1
        return False


@pytest.mark.asyncio
async def test_app_service_starts_channel_and_runs_inbound_and_outbox_loops() -> None:
    bus = MessageBus()
    channel = _Channel()
    queue = _Queue()
    bridge = _Bridge()
    outbox = _Outbox()
    service = AppService(
        channel=channel,
        bus=bus,
        bridge=bridge,
        queue=queue,
        outbox=outbox,
        clock=lambda: datetime(2026, 7, 14, tzinfo=UTC),
        idle_interval=0.001,
    )

    await service.start()
    await bus.publish_inbound(
        InboundMessage(channel="feishu", sender="ou-1", chat_id="chat-1", content="你好")
    )
    await asyncio.sleep(0.02)
    await service.stop()

    assert queue.ready is True
    assert channel.started is True
    assert channel.stopped is True
    assert [message.content for message in bridge.messages] == ["你好"]
    assert bus.inbound_size == 0
    assert outbox.calls > 0
