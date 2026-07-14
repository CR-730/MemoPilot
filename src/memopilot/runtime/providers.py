"""OpenAI-compatible Chat Provider。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from openai import AsyncOpenAI

from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    ToolSchema,
)


class ChatProvider(Protocol):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse: ...


class OpenAICompatibleProvider:
    """将 Runtime 合同适配为非流式 Chat Completions 请求。"""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_output_tokens: int = 2048,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._extra_body = dict(extra_body) if extra_body is not None else None

    @classmethod
    def from_credentials(
        cls,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_output_tokens: int = 2048,
        max_retries: int = 2,
        timeout_seconds: float = 60,
        extra_body: Mapping[str, Any] | None = None,
    ) -> OpenAICompatibleProvider:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout_seconds,
        )
        return cls(
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
            extra_body=extra_body,
        )

    @classmethod
    def from_deepseek_credentials(
        cls,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_output_tokens: int = 2048,
        max_retries: int = 2,
        timeout_seconds: float = 60,
    ) -> OpenAICompatibleProvider:
        """创建阶段 2 的 DeepSeek 非思考模式 Provider。

        DeepSeek V4 默认开启思考模式；思考模式下的工具调用要求回传
        ``reasoning_content``。阶段 2 暂不保存思维链，因此在适配边界显式关闭。
        """
        return cls.from_credentials(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            extra_body={"thinking": {"type": "disabled"}},
        )

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_openai() for message in messages],
            "max_tokens": self._max_output_tokens,
            "stream": False,
        }
        if tools:
            request["tools"] = tuple(tools)
            request["tool_choice"] = "auto"
        if self._extra_body is not None:
            request["extra_body"] = self._extra_body
        response = await self._client.chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("Provider 返回空 choices")
        choice = response.choices[0]
        message = choice.message
        calls = tuple(self._parse_call(call) for call in (message.tool_calls or ()))
        usage = getattr(response, "usage", None)
        return ModelResponse(
            content=message.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            response_id=getattr(response, "id", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    @staticmethod
    def _parse_call(raw_call: Any) -> FunctionCall:
        raw_arguments = raw_call.function.arguments or "{}"
        try:
            parsed = json.loads(raw_arguments)
            if not isinstance(parsed, dict):
                raise ValueError("工具参数必须是 JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return FunctionCall(
                id=raw_call.id,
                name=raw_call.function.name,
                arguments={},
                argument_error=f"工具参数不是有效 JSON object: {exc}",
            )
        return FunctionCall(
            id=raw_call.id,
            name=raw_call.function.name,
            arguments=parsed,
        )


__all__ = ["ChatProvider", "OpenAICompatibleProvider"]
