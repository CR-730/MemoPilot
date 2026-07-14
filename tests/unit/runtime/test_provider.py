from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from memopilot.runtime.contracts import ChatMessage, FunctionCall
from memopilot.runtime.providers import OpenAICompatibleProvider


class _Completions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _client(response: Any) -> tuple[Any, _Completions]:
    completions = _Completions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


async def test_provider_sends_tools_and_parses_function_calls() -> None:
    response = SimpleNamespace(
        id="chat-1",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="echo",
                                arguments=json.dumps({"text": "hello"}),
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
    )
    client, completions = _client(response)
    provider = OpenAICompatibleProvider(
        client=client,
        model="deepseek-v4-flash",
        max_output_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )
    tools = (
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo",
                "parameters": {"type": "object"},
            },
        },
    )

    result = await provider.complete(
        messages=(ChatMessage.user("hello"),),
        tools=tools,
    )

    assert result.tool_calls == (
        FunctionCall(id="call-1", name="echo", arguments={"text": "hello"}),
    )
    assert result.finish_reason == "tool_calls"
    assert result.prompt_tokens == 10
    assert completions.calls[0]["model"] == "deepseek-v4-flash"
    assert completions.calls[0]["tools"] == tools
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_provider_serializes_assistant_and_tool_messages() -> None:
    response = SimpleNamespace(
        id="chat-2",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="done", tool_calls=None),
            )
        ],
        usage=None,
    )
    client, completions = _client(response)
    provider = OpenAICompatibleProvider(client=client, model="model")
    call = FunctionCall(id="call-1", name="echo", arguments={"text": "hello"})

    result = await provider.complete(
        messages=(
            ChatMessage.user("hello"),
            ChatMessage.assistant(content=None, tool_calls=(call,)),
            ChatMessage.tool(call_id="call-1", name="echo", content="result"),
        ),
        tools=(),
    )

    assert result.content == "done"
    assert "tools" not in completions.calls[0]
    assert completions.calls[0]["messages"][1]["tool_calls"][0]["id"] == "call-1"
    assert completions.calls[0]["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "result",
    }


async def test_provider_preserves_invalid_tool_arguments_for_observation() -> None:
    response = SimpleNamespace(
        id="chat-3",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="bad",
                            function=SimpleNamespace(name="echo", arguments="{broken"),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    client, _ = _client(response)

    result = await OpenAICompatibleProvider(client=client, model="model").complete(
        messages=(ChatMessage.user("hello"),),
        tools=(),
    )

    assert result.tool_calls[0].arguments == {}
    assert result.tool_calls[0].argument_error is not None
