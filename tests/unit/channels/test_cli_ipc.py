from __future__ import annotations

import asyncio
import json

import pytest

from memopilot.channels.contracts import MessageBus
from memopilot.channels.ipc import IPCServerChannel


@pytest.mark.asyncio
async def test_ipc_server_copies_prototype_json_inbound_contract() -> None:
    probe = await asyncio.start_server(lambda _r, w: w.close(), "127.0.0.1", 0)
    port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()

    bus = MessageBus()
    received = asyncio.Future()

    async def capture(message: object) -> None:
        if not received.done():
            received.set_result(message)

    bus.subscribe_inbound(capture)
    channel = IPCServerChannel(bus, f"127.0.0.1:{port}")
    await channel.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write((json.dumps({"content": "你好"}, ensure_ascii=False) + "\n").encode())
    await writer.drain()
    message = await asyncio.wait_for(received, timeout=1)
    assert message.channel == "cli"
    assert message.content == "你好"
    writer.close()
    await writer.wait_closed()
    await channel.stop()
