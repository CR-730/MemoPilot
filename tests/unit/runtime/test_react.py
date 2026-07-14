from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    ToolSchema,
)
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.react import ReActEngine
from memopilot.runtime.tools import Tool, ToolRegistry


class _FakeProvider(ChatProvider):
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        self.calls.append({"messages": tuple(messages), "tools": tuple(tools)})
        return self._responses.pop(0)


class _FailingProvider(ChatProvider):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        raise TimeoutError("provider timeout")


def _response(
    content: str | None = None,
    calls: tuple[FunctionCall, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
    )


async def _echo(*, text: str) -> str:
    return f"echo:{text}"


async def _fail(*, text: str) -> str:
    raise RuntimeError(f"failed:{text}")


def _registry(handler: Any = _echo) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                handler=handler,
            )
        ]
    )


async def test_react_returns_direct_natural_language_reply() -> None:
    provider = _FakeProvider([_response("直接回复")])

    result = await ReActEngine(provider, _registry()).run(
        (ChatMessage.user("你好"),)
    )

    assert result.reply == "直接回复"
    assert result.exit_reason == "completed"
    assert result.iterations == 1
    assert result.tool_chain == ()


async def test_react_executes_function_and_returns_observation_to_model() -> None:
    call = FunctionCall(id="c1", name="echo", arguments={"text": "hello"})
    provider = _FakeProvider([_response(calls=(call,)), _response("工具调用完成")])

    result = await ReActEngine(provider, _registry()).run(
        (ChatMessage.user("回显 hello"),)
    )

    assert result.reply == "工具调用完成"
    assert result.iterations == 2
    assert result.tool_chain[0].observation.ok is True
    second_messages = provider.calls[1]["messages"]
    assert second_messages[-2].role == "assistant"
    assert second_messages[-2].tool_calls == (call,)
    assert second_messages[-1].role == "tool"
    assert second_messages[-1].tool_call_id == "c1"


@pytest.mark.parametrize(
    "call",
    [
        FunctionCall(id="missing", name="missing", arguments={}),
        FunctionCall(id="invalid", name="echo", arguments={}),
        FunctionCall(id="failed", name="echo", arguments={"text": "x"}),
    ],
)
async def test_react_returns_tool_failures_to_llm_for_natural_reply(
    call: FunctionCall,
) -> None:
    handler = _fail if call.id == "failed" else _echo
    provider = _FakeProvider([_response(calls=(call,)), _response("我已解释失败")])

    result = await ReActEngine(provider, _registry(handler)).run(
        (ChatMessage.user("执行"),)
    )

    assert result.reply == "我已解释失败"
    assert result.tool_chain[0].observation.ok is False
    observation_message = provider.calls[1]["messages"][-1]
    assert observation_message.role == "tool"
    assert '"ok": false' in (observation_message.content or "")


async def test_react_forces_tool_free_natural_language_summary_at_limit() -> None:
    call = FunctionCall(id="c1", name="echo", arguments={"text": "hello"})
    provider = _FakeProvider(
        [
            _response(calls=(call,)),
            _response("目前已取得工具结果，任务先在这里收尾。"),
        ]
    )

    result = await ReActEngine(provider, _registry(), max_iterations=1).run(
        (ChatMessage.user("执行复杂任务"),)
    )

    assert result.reply.startswith("目前")
    assert result.exit_reason == "max_iterations"
    assert len(provider.calls) == 2
    assert provider.calls[-1]["tools"] == ()
    assert provider.calls[-1]["messages"][-1].role == "system"


async def test_react_executes_multiple_calls_and_closes_the_tool_chain() -> None:
    calls = (
        FunctionCall(id="c1", name="echo", arguments={"text": "a"}),
        FunctionCall(id="c2", name="echo", arguments={"text": "b"}),
    )
    provider = _FakeProvider([_response(calls=calls), _response("完成")])

    result = await ReActEngine(provider, _registry()).run(
        (ChatMessage.user("执行两个工具"),)
    )

    assert [item.call.id for item in result.tool_chain] == ["c1", "c2"]
    assert [message.tool_call_id for message in provider.calls[1]["messages"][-2:]] == [
        "c1",
        "c2",
    ]


async def test_provider_failure_returns_explicit_degraded_result() -> None:
    result = await ReActEngine(_FailingProvider(), _registry()).run(
        (ChatMessage.user("hello"),)
    )

    assert result.exit_reason == "provider_error"
    assert result.infrastructure_error == "TimeoutError"
    assert result.reply
