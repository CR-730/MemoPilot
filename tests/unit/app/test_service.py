from __future__ import annotations

import pytest

from memopilot.app.service import AppService
from memopilot.channels.contracts import MessageBus


class _Channel:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_app_service_only_owns_channel_and_outbound_dispatch() -> None:
    bus = MessageBus()
    channel = _Channel()
    service = AppService(channel=channel, bus=bus)

    await service.start()
    await service.stop()

    assert channel.started is True
    assert channel.stopped is True
