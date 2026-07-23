"""主动内容候选与正文获取适配器。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from memopilot.runtime.tools import Tool


@dataclass(frozen=True, slots=True)
class ContentCandidate:
    item_id: str
    title: str
    source_id: str
    published_at: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("候选 item_id 不能为空")


class ContentFetcher(Protocol):
    async def fetch(self, url: str) -> str: ...


class ToolContentFetcher:
    """把 ToolRegistry 中的共享 ``web_fetch`` 适配为 Content 正文获取器。"""

    def __init__(self, tool: Tool, *, max_chars: int = 8_000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")
        self._tool = tool
        self._max_chars = max_chars

    async def fetch(self, url: str) -> str:
        async with asyncio.timeout(self._tool.timeout_seconds):
            result = await self._tool.handler(url=url, format="text")
        text = _tool_result_text(result)
        if not text.strip():
            raise ValueError("web_fetch 未返回正文")
        return text[: self._max_chars]


def _tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        try:
            return _tool_result_text(json.loads(result))
        except json.JSONDecodeError:
            return result
    if not isinstance(result, Mapping):
        return ""
    if result.get("error"):
        raise ValueError(str(result["error"]))
    if isinstance(result.get("text"), str):
        return str(result["text"])
    structured = result.get("structured")
    if isinstance(structured, Mapping):
        nested = _tool_result_text(structured)
        if nested:
            return nested
    content = result.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        texts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        if any(texts):
            return "\n".join(text for text in texts if text)
    return str(result.get("preview") or "")


__all__ = ["ContentCandidate", "ContentFetcher", "ToolContentFetcher"]
