from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _empty_media() -> list[str]:
    return []


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass
class InboundMessage:
    """从 Channel 传入的消息。"""

    channel: str
    sender: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=_empty_media)
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    @property
    def session_key(self) -> str:
        override = str(self.metadata.get("session_key_override") or "").strip()
        if override:
            return override
        return f"{self.channel}:{self.chat_id}"

    @property
    def context_channel(self) -> str:
        return str(self.metadata.get("context_channel") or self.channel).strip()

    @property
    def context_chat_id(self) -> str:
        return str(self.metadata.get("context_chat_id") or self.chat_id).strip()


@dataclass(frozen=True, slots=True)
class TurnCommitted:
    session_key: str
    channel: str
    chat_id: str
    input_message: str
    assistant_response: str
    tools_used: list[str]
    timestamp: datetime
    tool_chain: tuple[dict[str, str], ...] = ()
    react_cache_prompt_tokens: int = 0
    react_cache_hit_tokens: int = 0


__all__ = ["InboundMessage", "TurnCommitted"]
