"""从原型迁移的本地 CLI IPC 通道，负责接收输入并向客户端推送回复。"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from memopilot.channels.contracts import InboundMessage, MessageBus, SendReceipt

logger = logging.getLogger(__name__)


class IPCServerChannel:
    def __init__(self, bus: MessageBus, endpoint: str = "127.0.0.1:8765") -> None:
        self._bus = bus
        self._endpoint = endpoint
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        host, port = self._endpoint.rsplit(":", 1)
        self._server = await asyncio.start_server(self._handle_connection, host, int(port))
        logger.info("本地 CLI 已监听 tcp://%s", self._endpoint)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        for writer in tuple(self._writers.values()):
            writer.close()
        self._writers.clear()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        chat_id = f"cli-{uuid4().hex}"
        self._writers[chat_id] = writer
        try:
            while line := await reader.readline():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = str(payload.get("content", "")).strip()
                if not content:
                    continue
                await self._bus.publish_inbound(
                    InboundMessage(
                        channel="cli",
                        sender="cli-user",
                        chat_id=chat_id,
                        content=content,
                        metadata={"event_id": uuid4().hex, "message_id": uuid4().hex},
                    )
                )
        finally:
            self._writers.pop(chat_id, None)
            writer.close()
            await writer.wait_closed()

    async def send(self, chat_id: str, message: str, *, provider_uuid: str) -> SendReceipt:
        writer = self._writers.get(chat_id)
        if writer is None or writer.is_closing():
            raise ConnectionError("CLI 客户端已断开")
        writer.write((json.dumps({"content": message}, ensure_ascii=False) + "\n").encode())
        await writer.drain()
        return SendReceipt(message_id=provider_uuid)


__all__ = ["IPCServerChannel"]
