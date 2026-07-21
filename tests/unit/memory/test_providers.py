from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from memopilot.memory.providers import (
    ChatConsolidationExtractor,
    ChatHypothesisProvider,
    ChatImplicitMemoryExtractor,
    ChatOptimizerModel,
    ChatProcedureTagger,
    ChatRecentContextCompressor,
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
async def test_consolidation_and_implicit_long_term_use_separate_contracts() -> None:
    provider = _Provider(
        [
            json.dumps(
                {
                    "history_entries": [
                        {
                            "summary": "[2026-07-18 09:00] 用户开始修正记忆链路。",
                            "emotional_weight": 4,
                        }
                    ],
                    "pending_items": ["- [preference] 用户偏好先读原型生产代码。"],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "profile": [
                        {
                            "summary": "用户从事 AI Agent 开发",
                            "category": "personal_fact",
                            "emotional_weight": 0,
                        }
                    ],
                    "preference": [],
                    "procedure": [
                        {
                            "summary": "实现前先阅读原型生产代码",
                            "steps": ["定位实际入口", "阅读调用方和测试"],
                            "emotional_weight": 0,
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )

    archive = await ChatConsolidationExtractor(provider).extract("USER: 开始")  # type: ignore[arg-type]
    implicit = await ChatImplicitMemoryExtractor(provider).extract("USER: 开始")  # type: ignore[arg-type]

    assert archive["history_entries"][0]["emotional_weight"] == 4
    assert "memories" not in archive
    assert [item["kind"] for item in implicit] == ["profile", "procedure"]
    assert "6 个月后" in (provider.calls[1][1].content or "")
    assert "绝对不输出：event" in (provider.calls[1][1].content or "")


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
async def test_consolidation_normalizes_type_specific_metadata() -> None:
    provider = _Provider(
        [
            json.dumps(
                {
                    "artifacts": {},
                    "memories": [
                        {
                            "kind": "profile",
                            "summary": "用户目前在上海工作",
                            "category": "status",
                            "happened_at": "2026-07-17T10:00:00+00:00",
                        },
                        {
                            "kind": "procedure",
                            "summary": "删除文件前先移动到恢复目录",
                            "tool_requirement": "shell",
                            "steps": ["检查路径", "移动文件"],
                            "rule_schema": {
                                "required_tools": ["shell"],
                                "forbidden_tools": [],
                                "mentioned_tools": ["shell"],
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            )
        ]
    )

    result = await ChatConsolidationExtractor(provider).extract("USER: 内容")  # type: ignore[arg-type]
    memories = result["memories"]
    assert isinstance(memories, list)
    assert memories[0]["extra"] == {"category": "status"}
    assert memories[0]["happened_at"] == "2026-07-17T10:00:00+00:00"
    assert memories[1]["extra"] == {
        "tool_requirement": "shell",
        "steps": ["检查路径", "移动文件"],
        "rule_schema": {
            "required_tools": ["shell"],
            "forbidden_tools": [],
            "mentioned_tools": ["shell"],
        },
    }


@pytest.mark.asyncio
async def test_consolidation_downgrades_procedure_without_execution_condition() -> None:
    provider = _Provider(
        [
            '{"artifacts":{},"memories":['
            '{"kind":"procedure","summary":"用户喜欢简洁回答","steps":[]}'
            "]}"
        ]
    )

    result = await ChatConsolidationExtractor(provider).extract("USER: 简洁一点")  # type: ignore[arg-type]
    memories = result["memories"]
    assert isinstance(memories, list)
    assert memories[0]["kind"] == "preference"
    assert memories[0]["extra"] == {}


@pytest.mark.asyncio
async def test_procedure_tagger_filters_unknown_capabilities_and_normalizes_scope() -> None:
    provider = _Provider(
        [
            json.dumps(
                {
                    "scope": "global",
                    "tools": ["send_email", "invented_tool"],
                    "skills": ["invented_skill"],
                    "keywords": ["邮件"],
                },
                ensure_ascii=False,
            )
        ]
    )
    tagger = ChatProcedureTagger(
        provider,  # type: ignore[arg-type]
        allowed_tools={"send_email"},
        allowed_skills=set(),
    )

    result = await tagger.tag(
        "发送邮件前先确认",
        tool_requirement="send_email",
        steps=["展示草稿"],
    )

    assert result == {
        "scope": "tool_triggered",
        "tools": ["send_email"],
        "skills": [],
        "keywords": ["邮件"],
    }


@pytest.mark.asyncio
async def test_recent_context_compressor_uses_independent_structured_call() -> None:
    provider = _Provider(
        [
            json.dumps(
                {
                    "active_topics": ["阶段四记忆对齐"],
                    "user_preferences": ["希望实现前先验收设计"],
                    "follow_ups": ["继续修复 Procedure 生命周期"],
                    "avoidances": [],
                    "ongoing_threads": ["用户正在准备 AI Agent 求职项目"],
                },
                ensure_ascii=False,
            )
        ]
    )

    result = await ChatRecentContextCompressor(provider).compress(  # type: ignore[arg-type]
        old_context="# Recent Context\n\n## Compression\n- none",
        conversation="[2026-07-17] USER: 先计划和验收",
        recent_turns="[user] 开始修复",
        compression_until="2026-07-17T10:00:00+00:00",
    )

    assert result.startswith("# Recent Context")
    assert "## Compression" in result
    assert "- 最近持续关注：阶段四记忆对齐" in result
    assert "## Ongoing Threads" in result
    assert "- 用户正在准备 AI Agent 求职项目" in result
    assert "## Recent Turns" in result
    assert "[user] 开始修复" in result
    assert len(provider.calls) == 1


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


@pytest.mark.asyncio
async def test_consolidation_rejects_invalid_happened_at() -> None:
    provider = _Provider(
        [
            json.dumps(
                {
                    "artifacts": {},
                    "memories": [
                        {
                            "kind": "event",
                            "summary": "无效时间事件",
                            "happened_at": "2026-99-99",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ]
    )

    with pytest.raises(ValueError, match="ISO-8601"):
        await ChatConsolidationExtractor(provider).extract("USER: 内容")  # type: ignore[arg-type]
