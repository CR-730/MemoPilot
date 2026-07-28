from __future__ import annotations

import asyncio
import json

import pytest

import memopilot.channels.cli as cli
from memopilot.channels.ipc import IPCServerChannel


@pytest.mark.asyncio
async def test_ipc_server_copies_prototype_json_inbound_contract() -> None:
    probe = await asyncio.start_server(lambda _r, w: w.close(), "127.0.0.1", 0)
    port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()

    received = asyncio.Future()

    async def capture(message: object) -> None:
        if not received.done():
            received.set_result(message)

    channel = IPCServerChannel(capture, f"127.0.0.1:{port}")
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


@pytest.mark.asyncio
async def test_cli_client_can_send_and_receive_reply_offline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        assert "本地离线测试".encode() in line
        writer.write('{"content":"本地假服务回复"}\n'.encode())
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    lines = iter(["本地离线测试", "exit"])

    async def fake_read_line() -> str:
        value = next(lines)
        await asyncio.sleep(0.03)
        return value

    monkeypatch.setattr(cli, "_read_line", fake_read_line)
    await cli.CLIClient(f"127.0.0.1:{port}").run()
    await asyncio.sleep(0.08)
    server.close()
    await server.wait_closed()
    assert "本地假服务回复" in capsys.readouterr().out
