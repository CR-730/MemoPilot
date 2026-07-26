"""OpenAI-compatible Chat Provider。"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from openai import AsyncOpenAI

from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    StreamDelta,
    ToolSchema,
)


class ChatProvider(Protocol):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse: ...


class VisionProvider(Protocol):
    async def complete_vision(self, *, data_uri: str, prompt: str) -> str: ...


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
        return await self._complete_request(
            messages=messages,
            tools=tools,
            max_output_tokens=self._max_output_tokens,
            thinking_enabled=None,
        )

    async def complete_task(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
        max_output_tokens: int,
        thinking_enabled: bool | None = None,
    ) -> ModelResponse:
        return await self._complete_request(
            messages=messages,
            tools=tools,
            max_output_tokens=max_output_tokens,
            thinking_enabled=thinking_enabled,
        )

    async def complete_vision(self, *, data_uri: str, prompt: str) -> str:
        """使用当前 Provider 的模型完成一次独立视觉理解请求。"""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "high"},
                        },
                    ],
                }
            ],
            tools=(),
            max_tokens=self._max_output_tokens,
            stream=False,
        )
        if not response.choices:
            raise RuntimeError("视觉 Provider 返回空 choices")
        content = response.choices[0].message.content
        return str(content or "").strip()

    async def _complete_request(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
        max_output_tokens: int,
        thinking_enabled: bool | None,
    ) -> ModelResponse:
        preserve_reasoning = (
            self._preserve_reasoning_content
            if thinking_enabled is None
            else thinking_enabled
        )
        request_messages = [
            message.to_openai(
                include_provider_fields=preserve_reasoning,
            )
            for message in messages
        ]
        if preserve_reasoning:
            for message in request_messages:
                if message.get("role") == "assistant":
                    message.setdefault("reasoning_content", "")
        request: dict[str, Any] = {
            "model": self._model,
            "messages": request_messages,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if tools:
            request["tools"] = tuple(tools)
            request["tool_choice"] = "auto"
        extra_body = dict(self._extra_body) if self._extra_body is not None else None
        if thinking_enabled is not None:
            extra_body = extra_body or {}
            extra_body["thinking"] = {
                "type": "enabled" if thinking_enabled else "disabled"
            }
        if extra_body is not None:
            request["extra_body"] = extra_body
        response = await self._client.chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("Provider 返回空 choices")
        choice = response.choices[0]
        message = choice.message
        calls = tuple(self._parse_call(call) for call in (message.tool_calls or ()))
        thinking = (
            getattr(message, "reasoning_content", None)
            if preserve_reasoning
            else None
        )
        provider_fields: dict[str, Any] = {}
        if preserve_reasoning and (thinking is not None or calls):
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

    async def complete_streaming(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
        on_delta: Callable[[StreamDelta], Awaitable[None] | None],
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
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = tuple(tools)
            request["tool_choice"] = "auto"
        if self._extra_body is not None:
            request["extra_body"] = self._extra_body

        stream = await self._client.chat.completions.create(**request)
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_chunks: dict[int, dict[str, str]] = {}
        tool_call_seen = False
        response_id: str | None = None
        finish_reason: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        async for chunk in stream:
            response_id = str(getattr(chunk, "id", "") or response_id or "") or None
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
            choices = getattr(chunk, "choices", None) or ()
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            thinking_piece = getattr(delta, "reasoning_content", None)
            if isinstance(thinking_piece, str) and thinking_piece:
                thinking_parts.append(thinking_piece)
                if not tool_call_seen:
                    await _emit_delta(on_delta, StreamDelta(thinking_delta=thinking_piece))

            for raw_call in getattr(delta, "tool_calls", None) or ():
                tool_call_seen = True
                index = int(getattr(raw_call, "index", 0) or 0)
                slot = tool_chunks.setdefault(index, {"id": "", "name": "", "arguments": ""})
                function = getattr(raw_call, "function", None)
                slot["id"] += str(getattr(raw_call, "id", "") or "")
                slot["name"] += str(getattr(function, "name", "") or "")
                slot["arguments"] += str(getattr(function, "arguments", "") or "")

            content_piece = getattr(delta, "content", None)
            if isinstance(content_piece, str) and content_piece:
                content_parts.append(content_piece)
                if not tool_call_seen:
                    await _emit_delta(on_delta, StreamDelta(content_delta=content_piece))

        calls = tuple(
            self._parse_streamed_call(tool_chunks[index]) for index in sorted(tool_chunks)
        )
        thinking = "".join(thinking_parts).strip() or None
        provider_fields: dict[str, Any] = {}
        if self._preserve_reasoning_content and (thinking is not None or calls):
            provider_fields["reasoning_content"] = thinking or ""
        return ModelResponse(
            content="".join(content_parts).strip() or None,
            tool_calls=calls,
            finish_reason=finish_reason,
            response_id=response_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking=thinking,
            provider_fields=provider_fields,
        )

    @staticmethod
    def _parse_streamed_call(raw_call: Mapping[str, str]) -> FunctionCall:
        raw_arguments = raw_call.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_arguments)
            if not isinstance(parsed, dict):
                raise ValueError("工具参数必须是 JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return FunctionCall(
                id=raw_call.get("id", ""),
                name=raw_call.get("name", ""),
                arguments={},
                argument_error=f"工具参数不是有效 JSON object: {exc}",
            )
        return FunctionCall(
            id=raw_call.get("id", ""),
            name=raw_call.get("name", ""),
            arguments=parsed,
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


async def _emit_delta(
    callback: Callable[[StreamDelta], Awaitable[None] | None],
    delta: StreamDelta,
) -> None:
    result = callback(delta)
    if inspect.isawaitable(result):
        await result


__all__ = ["ChatProvider", "OpenAICompatibleProvider", "VisionProvider"]
