from __future__ import annotations

import json
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
            '```json\n{"artifacts":{"PENDING.md":"- [preference] 中文"},"memories":[]}\n```',
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
    provider = _Provider(
        [
            "事件假设",
            "# 用户长期记忆\n\n## 用户事实\n- 新事实\n\n## 用户偏好\n- 暂无\n\n"
            "## 用户明确要求长期记住的关键内容\n- 暂无",
            "# MemoPilot 的自我认知\n\n## 人格与形象\n- 稳定\n\n"
            "## 我对当前用户的理解\n- 尊重事实\n\n## 我们关系的定义\n- 长期协作",
        ]
    )

    hypothesis = await ChatHypothesisProvider(provider).generate(  # type: ignore[arg-type]
        "原问题", style="event"
    )
    memory, self_text = await ChatOptimizerModel(provider).optimize(  # type: ignore[arg-type]
        "旧记忆", "旧自我", "新事实"
    )

    assert hypothesis == "事件假设"
    assert memory.startswith("# 用户长期记忆")
    assert self_text.startswith("# MemoPilot 的自我认知")
    assert all(call[0].role == "system" for call in provider.calls)
    assert "缺席成本测试" in provider.calls[1][1].content
    assert "只允许保留以下三个 section" in provider.calls[2][1].content


@pytest.mark.asyncio
async def test_optimizer_rejects_unknown_or_missing_markdown_sections() -> None:
    provider = _Provider(
        [
            "# 用户长期记忆\n\n## 用户事实\n- 事实\n\n## 用户偏好\n- 偏好\n\n"
            "## 用户明确要求长期记住的关键内容\n- 暂无\n\n## 调试日志\n- 不允许",
            "# MemoPilot 的自我认知\n\n## 人格与形象\n- 稳定",
        ]
    )
    model = ChatOptimizerModel(provider)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="MEMORY.md"):
        await model.optimize("旧记忆", "旧自我", "新事实")


@pytest.mark.asyncio
async def test_consolidation_rejects_invalid_memory_kind() -> None:
    provider = _Provider(['{"artifacts":{},"memories":[{"kind":"guess","summary":"未经证实"}]}'])

    with pytest.raises(ValueError, match="kind"):
        await ChatConsolidationExtractor(provider).extract("USER: 也许")  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending",
    [
        "- [procedure] 以后先搜索再回答",
        "没有标签的正文",
        "## 用户偏好\n- [preference] 中文",
        "- [unknown] 未知标签",
    ],
)
async def test_consolidation_rejects_invalid_pending_lines(pending: str) -> None:
    provider = _Provider(
        [json.dumps({"artifacts": {"PENDING.md": pending}, "memories": []}, ensure_ascii=False)]
    )

    with pytest.raises(ValueError, match="PENDING.md"):
        await ChatConsolidationExtractor(provider).extract("USER: 内容")  # type: ignore[arg-type]
