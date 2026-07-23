"""旧原型的主动消息语义去重器。检测失败时放行，不阻断主动链路。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.providers import ChatProvider


class MessageDeduper:
    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    async def is_duplicate(
        self, new_message: str, recent_proactive: Sequence[str]
    ) -> tuple[bool, str]:
        if not recent_proactive:
            return False, "无近期主动消息，放行"
        try:
            response = await self._provider.complete(
                messages=(
                    ChatMessage.system(
                        "你是消息重复检测器。判断新消息是否与近期已发主动消息在实质信息上"
                        "重复；同话题有真正新进展不算重复。只输出 JSON。"
                    ),
                    ChatMessage.user(
                        "近期已发消息：\n"
                        + "\n---\n".join(recent_proactive)
                        + f"\n\n新消息：\n{new_message}\n\n"
                        + '只输出 {"is_duplicate": false, "reason": "简短说明"}'
                    ),
                ),
                tools=(),
            )
            payload = _json_object(response.content or "")
            return bool(payload.get("is_duplicate", False)), str(payload.get("reason") or "")
        except Exception as exc:
            return False, str(exc)


def _json_object(text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("语义去重模型没有返回 JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("语义去重结果必须是对象")
    return value


__all__ = ["MessageDeduper"]
