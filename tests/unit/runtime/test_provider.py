from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.runtime.contracts import ChatMessage, FunctionCall, StreamDelta
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


async def test_provider_round_trips_deepseek_reasoning_content() -> None:
    response = SimpleNamespace(
        id="chat-thinking",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    reasoning_content="先查资料",
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
        usage=None,
    )
    client, completions = _client(response)
    provider = OpenAICompatibleProvider(
        client=client,
        model="deepseek-v4-pro",
        extra_body={"thinking": {"type": "enabled"}},
        preserve_reasoning_content=True,
    )

    result = await provider.complete(
        messages=(ChatMessage.user("hello"),),
        tools=(),
    )
    assistant = ChatMessage.assistant(
        content=result.content,
        tool_calls=result.tool_calls,
        provider_fields=result.provider_fields,
    )
    await provider.complete(messages=(assistant,), tools=())

    assert result.thinking == "先查资料"
    assert result.provider_fields == {"reasoning_content": "先查资料"}
    assert completions.calls[1]["messages"][0]["reasoning_content"] == "先查资料"


async def test_deepseek_factory_enables_thinking_and_fills_missing_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        id="chat-thinking-empty",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name="echo", arguments="{}"),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    client, completions = _client(response)
    monkeypatch.setattr("memopilot.runtime.providers.AsyncOpenAI", lambda **_: client)
    provider = OpenAICompatibleProvider.from_deepseek_credentials(
        api_key="test",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        thinking_enabled=True,
    )

    result = await provider.complete(
        messages=(
            ChatMessage.user("hello"),
            ChatMessage.assistant(content="旧回复"),
            ChatMessage.user("继续"),
        ),
        tools=(),
    )

    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert completions.calls[0]["messages"][1]["reasoning_content"] == ""
    assert result.provider_fields == {"reasoning_content": ""}


async def test_generic_provider_strips_deepseek_provider_fields() -> None:
    response = SimpleNamespace(
        id="chat-generic",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="done", tool_calls=None),
            )
        ],
        usage=None,
    )
    client, completions = _client(response)
    provider = OpenAICompatibleProvider(client=client, model="generic-model")
    assistant = ChatMessage.assistant(
        content="旧回复",
        provider_fields={"reasoning_content": "不应发送"},
    )

    await provider.complete(messages=(assistant,), tools=())

    assert "reasoning_content" not in completions.calls[0]["messages"][0]


async def test_deepseek_factory_disables_thinking_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        id="chat-default",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="done",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )
    client, completions = _client(response)
    monkeypatch.setattr("memopilot.runtime.providers.AsyncOpenAI", lambda **_: client)
    provider = OpenAICompatibleProvider.from_deepseek_credentials(
        api_key="test",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
    )

    await provider.complete(messages=(ChatMessage.user("hello"),), tools=())

    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


class _AsyncChunks:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


async def test_streaming_provider_emits_reasoning_and_content_deltas() -> None:
    stream = _AsyncChunks(
        [
            SimpleNamespace(
                id="chat-stream",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            reasoning_content="先分析",
                            content=None,
                            tool_calls=None,
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chat-stream",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            reasoning_content=None,
                            content="最终答案",
                            tool_calls=None,
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=5),
            ),
        ]
    )
    client, completions = _client(stream)
    provider = OpenAICompatibleProvider(
        client=client,
        model="deepseek-v4-pro",
        preserve_reasoning_content=True,
    )
    deltas: list[StreamDelta] = []

    result = await provider.complete_streaming(
        messages=(ChatMessage.user("hello"),),
        tools=(),
        on_delta=deltas.append,
    )

    assert deltas == [
        StreamDelta(thinking_delta="先分析"),
        StreamDelta(content_delta="最终答案"),
    ]
    assert result.thinking == "先分析"
    assert result.content == "最终答案"
    assert result.provider_fields == {"reasoning_content": "先分析"}
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 8
    assert result.completion_tokens == 5
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["stream_options"] == {"include_usage": True}


async def test_streaming_provider_reassembles_tool_call_chunks() -> None:
    stream = _AsyncChunks(
        [
            SimpleNamespace(
                id="chat-tool-stream",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            reasoning_content="先查",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-",
                                    function=SimpleNamespace(name="echo", arguments='{"text":'),
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="chat-tool-stream",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            reasoning_content=None,
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="1",
                                    function=SimpleNamespace(name="", arguments='"hi"}'),
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            ),
        ]
    )
    client, _ = _client(stream)
    provider = OpenAICompatibleProvider(
        client=client,
        model="deepseek-v4-pro",
        preserve_reasoning_content=True,
    )
    deltas: list[StreamDelta] = []

    result = await provider.complete_streaming(
        messages=(ChatMessage.user("hello"),),
        tools=(),
        on_delta=deltas.append,
    )

    assert result.tool_calls == (
        FunctionCall(id="call-1", name="echo", arguments={"text": "hi"}),
    )
    assert result.provider_fields == {"reasoning_content": "先查"}
    assert deltas == [StreamDelta(thinking_delta="先查")]
