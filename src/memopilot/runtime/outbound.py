from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.runtime.common_tools.message_push import MessagePushTool

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@dataclass
class OutboundDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    media: list[str] = field(default_factory=list)


class OutboundPort(Protocol):
    async def dispatch(self, outbound: OutboundDispatch) -> bool: ...


class DeliveryError(RuntimeError):
    pass


class PushToolOutboundPort:
    """通过公共 message_push 工具发送后台消息。"""

    def __init__(self, message_push: MessagePushTool) -> None:
        self._message_push = message_push

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        results: list[str] = []
        if outbound.content:
            results.append(
                await self._message_push.execute(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    message=outbound.content,
                    provider_uuid=outbound.metadata.get("provider_uuid"),
                )
            )
        base_uuid = str(outbound.metadata.get("provider_uuid") or "")
        if not base_uuid:
            base_uuid = "\0".join(
                (
                    outbound.channel,
                    outbound.chat_id,
                    outbound.content,
                    *outbound.media,
                )
            )
        for index, media in enumerate(outbound.media):
            field = "image" if Path(media).suffix.lower() in _IMAGE_SUFFIXES else "file"
            media_uuid = str(
                uuid5(
                    NAMESPACE_URL,
                    f"memopilot:media:{base_uuid}:{index}:{media}",
                )
            )
            results.append(
                await self._message_push.execute(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    provider_uuid=media_uuid,
                    **{field: media},
                )
            )
        return bool(results) and all("已发送" in result for result in results)


__all__ = [
    "DeliveryError",
    "OutboundDispatch",
    "OutboundPort",
    "PushToolOutboundPort",
]
