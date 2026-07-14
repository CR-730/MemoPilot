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
        preserve_reasoning_content: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._extra_body = dict(extra_body) if extra_body is not None else None
        self._preserve_reasoning_content = preserve_reasoning_content

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
        preserve_reasoning_content: bool = False,
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
            preserve_reasoning_content=preserve_reasoning_content,
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
        thinking_enabled: bool = False,
    ) -> OpenAICompatibleProvider:
        """创建保留思考字段的 DeepSeek Provider。"""
        return cls.from_credentials(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            extra_body={
                "thinking": {
                    "type": "enabled" if thinking_enabled else "disabled",
                }
            },
            preserve_reasoning_content=thinking_enabled,
        )

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        request_messages = [
            message.to_openai(
                include_provider_fields=self._preserve_reasoning_content,
            )
            for message in messages
        ]
        if self._preserve_reasoning_content:
            for message in request_messages:
                if message.get("role") == "assistant":
                    message.setdefault("reasoning_content", "")
        request: dict[str, Any] = {
            "model": self._model,
            "messages": request_messages,
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
        thinking = (
            getattr(message, "reasoning_content", None)
            if self._preserve_reasoning_content
            else None
        )
        provider_fields: dict[str, Any] = {}
        if self._preserve_reasoning_content and (thinking is not None or calls):
            provider_fields["reasoning_content"] = (
                str(thinking) if thinking is not None else ""
            )
        usage = getattr(response, "usage", None)
        return ModelResponse(
            content=message.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            response_id=getattr(response, "id", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            thinking=str(thinking) if thinking is not None else None,
            provider_fields=provider_fields,
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
