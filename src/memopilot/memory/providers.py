"""记忆模块与通用 Chat/Embedding Provider 的适配。"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from memopilot.memory.prompts import (
    CONSOLIDATION_SYSTEM,
    CONSOLIDATION_USER,
    HYPOTHESIS_SYSTEM,
    MEMORY_OPTIMIZER_SYSTEM,
    SELF_OPTIMIZER_SYSTEM,
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
        return _load_json_object(content)


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
                ChatMessage.user(f"现有 MEMORY.md：\n{memory}\n\n待合并事实：\n{pending}"),
            ),
            tools=(),
        )
        self_response = await self.provider.complete(
            messages=(
                ChatMessage.system(SELF_OPTIMIZER_SYSTEM),
                ChatMessage.user(
                    f"现有 SELF.md：\n{self_text}\n\n本轮待合并事实（仅作证据）：\n{pending}"
                ),
            ),
            tools=(),
        )
        memory_output = (memory_response.content or "").strip()
        self_output = (self_response.content or "").strip()
        if not memory_output or not self_output:
            raise ValueError("记忆优化模型返回空响应")
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


__all__ = [
    "ChatConsolidationExtractor",
    "ChatHypothesisProvider",
    "ChatOptimizerModel",
    "OpenAIEmbeddingProvider",
]
