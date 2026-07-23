from __future__ import annotations

from memopilot.proactive.dedupe import MessageDeduper
from memopilot.runtime.contracts import ModelResponse


class _Provider:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ModelResponse(self.content, ())


async def test_message_deduper_uses_old_prototype_json_contract() -> None:
    provider = _Provider('{"is_duplicate": true, "reason": "同一事件"}')

    duplicate, reason = await MessageDeduper(provider).is_duplicate("新消息", ("旧消息",))

    assert duplicate is True
    assert reason == "同一事件"


async def test_message_deduper_fails_open_and_skips_empty_history() -> None:
    provider = _Provider(error=RuntimeError("provider down"))
    deduper = MessageDeduper(provider)

    assert await deduper.is_duplicate("新消息", ()) == (False, "无近期主动消息，放行")
    duplicate, reason = await deduper.is_duplicate("新消息", ("旧消息",))

    assert duplicate is False
    assert "provider down" in reason
