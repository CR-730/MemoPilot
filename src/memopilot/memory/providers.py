"""记忆模块与通用 Chat/Embedding Provider 的适配。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from memopilot.memory.prompts import (
    CONSOLIDATION_SYSTEM,
    CONSOLIDATION_USER,
    HYPOTHESIS_SYSTEM,
    MEMORY_OPTIMIZER_SYSTEM,
    MEMORY_OPTIMIZER_USER,
    SELF_OPTIMIZER_SYSTEM,
    SELF_OPTIMIZER_USER,
)
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.providers import ChatProvider


class ChatConsolidationExtractor:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def extract(self, conversation: str) -> dict[str, object]:
        response = await self.provider.complete(
            messages=(
                ChatMessage.system(CONSOLIDATION_SYSTEM),
                ChatMessage.user(CONSOLIDATION_USER.format(conversation=conversation)),
            ),
            tools=(),
        )
        content = (response.content or "").strip()
        if not content:
            raise ValueError("Consolidation 模型返回空响应")
        return _validate_consolidation_output(_load_json_object(content))


class ChatHypothesisProvider:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def generate(self, query: str, *, style: str) -> str:
        focus = "可能发生过的具体事件" if style == "event" else "一般事实或稳定偏好"
        response = await self.provider.complete(
            messages=(
                ChatMessage.system(HYPOTHESIS_SYSTEM),
                ChatMessage.user(f"原问题：{query}\n改写重点：{focus}"),
            ),
            tools=(),
        )
        return (response.content or "").strip()


class ChatOptimizerModel:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def optimize(self, memory: str, self_text: str, pending: str) -> tuple[str, str]:
        memory_response = await self.provider.complete(
            messages=(
                ChatMessage.system(MEMORY_OPTIMIZER_SYSTEM),
                ChatMessage.user(
                    MEMORY_OPTIMIZER_USER.format(
                        today=datetime.now().strftime("%Y-%m-%d"),
                        memory=memory or "（空）",
                        pending=pending or "（无新内容）",
                    )
                ),
            ),
            tools=(),
        )
        self_response = await self.provider.complete(
            messages=(
                ChatMessage.system(SELF_OPTIMIZER_SYSTEM),
                ChatMessage.user(
                    SELF_OPTIMIZER_USER.format(
                        self_text=self_text or "（空）",
                        pending=pending or "（无新内容）",
                    )
                ),
            ),
            tools=(),
        )
        memory_output = (memory_response.content or "").strip()
        self_output = (self_response.content or "").strip()
        if not memory_output or not self_output:
            raise ValueError("记忆优化模型返回空响应")
        _validate_markdown_contract(
            memory_output,
            name="MEMORY.md",
            title="# 用户长期记忆",
            required_sections=(
                "## 用户事实",
                "## 用户偏好",
                "## 用户明确要求长期记住的关键内容",
            ),
            optional_sections=("## 助手操作上下文",),
        )
        _validate_markdown_contract(
            self_output,
            name="SELF.md",
            title="# MemoPilot 的自我认知",
            required_sections=(
                "## 人格与形象",
                "## 我对当前用户的理解",
                "## 我们关系的定义",
            ),
        )
        return memory_output, self_output


class OpenAIEmbeddingProvider:
    """兼容 OpenAI Embeddings API 的向量模型适配器。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(model=self.model, input=text)
        if not response.data:
            raise ValueError("Embedding Provider 返回空向量")
        return [float(value) for value in response.data[0].embedding]


def _load_json_object(text: str) -> dict[str, object]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    try:
        loaded: Any = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型响应不包含 JSON 对象") from None
        loaded = json.loads(value[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("模型响应必须是 JSON 对象")
    return {str(key): item for key, item in loaded.items()}


def _validate_consolidation_output(output: dict[str, object]) -> dict[str, object]:
    artifacts = output.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Consolidation artifacts 必须是对象")
    allowed_artifacts = {"HISTORY.md", "PENDING.md", "CONTEXT.md"}
    for name, content in artifacts.items():
        if str(name) not in allowed_artifacts or not isinstance(content, str):
            raise ValueError(f"Consolidation artifact 无效: {name}")
        if str(name) == "PENDING.md":
            _validate_pending_artifact(content)
    memories = output.get("memories", [])
    if not isinstance(memories, list):
        raise ValueError("Consolidation memories 必须是数组")
    allowed_kinds = {"event", "profile", "preference", "procedure"}
    for index, memory in enumerate(memories):
        if not isinstance(memory, dict):
            raise ValueError(f"Consolidation memories[{index}] 必须是对象")
        kind = memory.get("kind")
        summary = memory.get("summary")
        if kind not in allowed_kinds:
            raise ValueError(f"Consolidation memories[{index}].kind 无效: {kind}")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"Consolidation memories[{index}].summary 不能为空")
        weight = memory.get("emotional_weight", 0)
        if not isinstance(weight, int) or isinstance(weight, bool) or not 0 <= weight <= 10:
            raise ValueError(f"Consolidation memories[{index}].emotional_weight 无效")
    return output


def _validate_pending_artifact(content: str) -> None:
    allowed_tags = {
        "identity",
        "preference",
        "key_info",
        "health_long_term",
        "requested_memory",
        "correction",
        "agent_context",
    }
    for line_number, line in enumerate(content.splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        match = re.fullmatch(r"- \[([a-z_]+)]\s+(.+)", value)
        if match is None or match.group(1) not in allowed_tags:
            raise ValueError(f"PENDING.md 第 {line_number} 行标签或格式无效")


def _validate_markdown_contract(
    text: str,
    *,
    name: str,
    title: str,
    required_sections: tuple[str, ...],
    optional_sections: tuple[str, ...] = (),
) -> None:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    if not lines or lines[0] != title:
        raise ValueError(f"{name} 标题无效，必须是 {title}")
    headings = [line for line in lines if re.fullmatch(r"## .+", line)]
    allowed = (*required_sections, *optional_sections)
    if any(heading not in allowed for heading in headings):
        raise ValueError(f"{name} 包含不允许的 section")
    if tuple(headings[: len(required_sections)]) != required_sections:
        raise ValueError(f"{name} 缺少必需 section 或顺序错误")
    if len(headings) != len(set(headings)):
        raise ValueError(f"{name} section 不得重复")
    if headings[len(required_sections) :] not in ([], list(optional_sections)):
        raise ValueError(f"{name} 可选 section 顺序错误")
    for line in lines[1:]:
        if line and not line.startswith(("## ", "- ")):
            raise ValueError(f"{name} 只能包含标题、section 和 bullet")


__all__ = [
    "ChatConsolidationExtractor",
    "ChatHypothesisProvider",
    "ChatOptimizerModel",
    "OpenAIEmbeddingProvider",
]
