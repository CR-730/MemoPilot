"""原型插件回调使用的属性化事件 DTO。"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from memopilot.runtime.contracts import ChatMessage


class PluginEventContext(MutableMapping[str, object]):
    """同时支持 `event.field` 与旧字典访问的生命周期上下文。"""

    _payload: dict[str, object]

    def __init__(self, payload: dict[str, object]) -> None:
        object.__setattr__(self, "_payload", dict(payload))

    def __getitem__(self, key: str) -> object:
        return self._payload[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._payload[key] = value

    def __delitem__(self, key: str) -> None:
        del self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def __getattr__(self, key: str) -> object:
        try:
            return self._payload[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: object) -> None:
        self._payload[key] = value

    def as_dict(self) -> dict[str, object]:
        return dict(self._payload)


@dataclass(slots=True)
class PreToolCtx:
    tool_name: str
    arguments: dict[str, object]
    session_key: str | None = None
    channel: str | None = None
    chat_id: str | None = None
    call_id: str | None = None
    source: str | None = None
    request_text: str | None = None
    tool_batch: tuple[object, ...] = ()
    tool_batch_index: int | None = None


@dataclass(slots=True)
class BeforeTurnInput:
    """当前简化 Runtime 能真实承载的 turn 状态。

    原型的 ``msg/session`` 依赖持久会话对象；MemoPilot 当前阶段没有该对象，
    因而不伪造这两个字段。插件应使用这里的显式字段。
    """

    session_key: str
    channel: str
    chat_id: str
    content: str
    system_prompt: str
    history: tuple[ChatMessage, ...]
    prompt_scope: str
    media: tuple[str, ...] = ()
    outbound_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class BeforeTurnCtx:
    session_key: str
    channel: str
    chat_id: str
    content: str
    system_prompt: str
    history: tuple[ChatMessage, ...]
    prompt_scope: str
    media: tuple[str, ...] = ()
    extra_hints: list[str] = field(default_factory=list)
    outbound_metadata: dict[str, object] = field(default_factory=dict)
    abort: bool = False
    abort_reply: str = ""


@dataclass(frozen=True, slots=True)
class BeforeReasoningInput:
    state: BeforeTurnInput
    before_turn: BeforeTurnCtx


@dataclass(slots=True)
class BeforeReasoningCtx:
    session_key: str
    channel: str
    chat_id: str
    content: str
    media: tuple[str, ...]
    history: tuple[ChatMessage, ...]
    prompt_scope: str
    system_prompt: str
    extra_hints: list[str] = field(default_factory=list)
    visible_tool_names: frozenset[str] | None = None
    abort: bool = False
    abort_reply: str = ""


@dataclass(frozen=True, slots=True)
class PromptRenderInput:
    session_key: str
    channel: str
    chat_id: str
    content: str
    media: tuple[str, ...]
    history: tuple[ChatMessage, ...]
    prompt_scope: str
    system_prompt: str
    extra_hints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptRenderResult:
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class BeforeStepInput:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    messages: tuple[ChatMessage, ...]
    visible_names: frozenset[str] | None


@dataclass(slots=True)
class BeforeStepCtx:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    input_tokens_estimate: int
    visible_tool_names: frozenset[str] | None
    extra_hints: list[str] = field(default_factory=list)
    early_stop: bool = False
    early_stop_reply: str = ""


@dataclass(slots=True)
class AfterStepCtx:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    context_tokens_estimate: int
    tools_called: tuple[str, ...]
    partial_reply: str
    tools_used_so_far: tuple[str, ...]
    tool_chain_partial: tuple[dict[str, object], ...]
    partial_thinking: str | None
    has_more: bool
    early_stop: bool = False
    early_stop_reason: str = ""
    extra_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AfterReasoningInput:
    state: BeforeTurnInput
    turn_result: Any


@dataclass(slots=True)
class AfterReasoningCtx:
    session_key: str
    channel: str
    chat_id: str
    tools_used: tuple[str, ...]
    thinking: str | None
    tool_chain: tuple[dict[str, object], ...]
    reply: str
    media: list[str] = field(default_factory=list)
    outbound_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AfterReasoningResult:
    ctx: AfterReasoningCtx
    turn_result: Any


@dataclass(frozen=True, slots=True)
class AfterTurnInput:
    state: BeforeTurnInput
    result: AfterReasoningResult


@dataclass(slots=True)
class AfterTurnCtx:
    session_key: str
    channel: str
    chat_id: str
    reply: str
    tools_used: tuple[str, ...]
    thinking: str | None
    media: tuple[str, ...]
    outbound_metadata: dict[str, object]
    will_dispatch: bool = False
    extra_metadata: dict[str, object] = field(default_factory=dict)


__all__ = [
    "AfterReasoningCtx",
    "AfterReasoningInput",
    "AfterReasoningResult",
    "AfterStepCtx",
    "AfterTurnCtx",
    "AfterTurnInput",
    "BeforeReasoningCtx",
    "BeforeReasoningInput",
    "BeforeStepCtx",
    "BeforeStepInput",
    "BeforeTurnCtx",
    "BeforeTurnInput",
    "PluginEventContext",
    "PreToolCtx",
    "PromptRenderInput",
    "PromptRenderResult",
]
