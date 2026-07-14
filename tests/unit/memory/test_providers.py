from __future__ import annotations

from collections.abc import Sequence

import pytest

from memopilot.memory.providers import (
    ChatConsolidationExtractor,
    ChatHypothesisProvider,
    ChatOptimizerModel,
)
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema


class _Provider:
    def __init__(self, responses: list[str | None]) -> None:
        self.responses = responses
        self.calls: list[tuple[ChatMessage, ...]] = []

    async def complete(
        self, *, messages: Sequence[ChatMessage], tools: Sequence[ToolSchema]
    ) -> ModelResponse:
        assert tools == ()
        self.calls.append(tuple(messages))
        return ModelResponse(content=self.responses.pop(0), tool_calls=(), finish_reason="stop")


@pytest.mark.asyncio
async def test_consolidation_adapter_parses_fenced_json_and_rejects_empty_output() -> None:
    provider = _Provider(
        [
            '```json\n{"artifacts":{"PENDING.md":"- [preference] 中文"},'
            '"memories":[]}\n```',
            None,
        ]
    )
    extractor = ChatConsolidationExtractor(provider)  # type: ignore[arg-type]

    result = await extractor.extract("USER: 使用中文")
    assert result["artifacts"] == {"PENDING.md": "- [preference] 中文"}
    with pytest.raises(ValueError, match="空响应"):
        await extractor.extract("USER: hello")


@pytest.mark.asyncio
async def test_hypothesis_and_optimizer_adapters_keep_model_roles_explicit() -> None:
    provider = _Provider(["事件假设", "# 长期记忆\n- 新事实", "# MemoPilot\n稳定"])

    hypothesis = await ChatHypothesisProvider(provider).generate(  # type: ignore[arg-type]
        "原问题", style="event"
    )
    memory, self_text = await ChatOptimizerModel(provider).optimize(  # type: ignore[arg-type]
        "旧记忆", "旧自我", "新事实"
    )

    assert hypothesis == "事件假设"
    assert memory.startswith("# 长期记忆")
    assert self_text.startswith("# MemoPilot")
    assert all(call[0].role == "system" for call in provider.calls)
