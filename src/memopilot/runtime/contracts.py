"""Agent Runtime 内部使用的稳定类型合同。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

type ToolSchema = dict[str, Any]


@dataclass(frozen=True)
class StreamDelta:
    content_delta: str = ""
    thinking_delta: str = ""


@dataclass(frozen=True)
class FunctionCall:
    id: str
    name: str
    arguments: dict[str, Any]
    argument_error: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None
    tool_calls: tuple[FunctionCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    provider_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def user_blocks(cls, content: list[dict[str, Any]]) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        *,
        content: str | None,
        tool_calls: tuple[FunctionCall, ...] = (),
        provider_fields: Mapping[str, Any] | None = None,
    ) -> ChatMessage:
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            provider_fields=dict(provider_fields or {}),
        )

    @classmethod
    def tool(cls, *, call_id: str, name: str, content: str) -> ChatMessage:
        return cls(
            role="tool",
            content=content,
            tool_call_id=call_id,
            name=name,
        )

    def to_openai(self, *, include_provider_fields: bool = False) -> dict[str, Any]:
        if self.role == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": _json_arguments(call.arguments),
                        },
                    }
                    for call in self.tool_calls
                ]
            if include_provider_fields:
                message.update(self.provider_fields)
            return message
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool message 缺少 tool_call_id")
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or "",
            }
        return {"role": self.role, "content": self.content or ""}


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    tool_calls: tuple[FunctionCall, ...]
    finish_reason: str | None = None
    response_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thinking: str | None = None
    provider_fields: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


def _json_arguments(arguments: dict[str, Any]) -> str:
    import json

    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


__all__ = ["ChatMessage", "FunctionCall", "ModelResponse", "StreamDelta", "ToolSchema"]
