from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest

from memopilot.extensions.events import EventBus
from memopilot.extensions.hooks import ToolHook, ToolHookDecision
from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    StreamDelta,
    ToolSchema,
)
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.react import ReActEngine
from memopilot.runtime.tool_search import ToolSearchTool
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


async def test_react_logs_prototype_style_steps_and_token_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call = FunctionCall(id="c1", name="echo", arguments={"text": "hello"})
    provider = _FakeProvider(
        [
            ModelResponse(
                content=None,
                tool_calls=(call,),
                finish_reason="tool_calls",
                prompt_tokens=120,
                completion_tokens=8,
            ),
            ModelResponse(
                content="工具调用完成",
                tool_calls=(),
                finish_reason="stop",
                prompt_tokens=150,
                completion_tokens=12,
            ),
        ]
    )

    with caplog.at_level(logging.INFO, logger="memopilot.runtime.react"):
        await ReActEngine(
            provider,
            _registry(),
            session_key="feishu:chat-1",
        ).run((ChatMessage.user("回显 hello"),))

    messages = [record.getMessage() for record in caplog.records]
    assert any("[LLM调用] 第1轮" in message and "input_tokens~=" in message for message in messages)
    assert any("[LLM决策→工具] 第1轮，调用: ['echo']" in message for message in messages)
    assert any("[工具执行→] echo" in message and "hello" in message for message in messages)
    assert any("[工具结果←] echo" in message and "result_len=" in message for message in messages)
    assert any("[LLM调用] 第2轮" in message and "input_tokens~=" in message for message in messages)
    assert any("[LLM决策→回复] 第2轮，共调用工具1次: ['echo']" in message for message in messages)
    assert any(
        "react_context: session_key=feishu:chat-1"
        in message
        and "iteration_count=2"
        in message
        and "turn_input_sum_tokens~="
        in message
        and "turn_input_peak_tokens~="
        in message
        and "final_call_input_tokens~="
        in message
        and "prompt_tokens=270"
        in message
        and "completion_tokens=20"
        in message
        for message in messages
    )


async def test_react_appends_multimodal_tool_blocks_as_user_message() -> None:
    async def read_image() -> dict[str, object]:
        return {
            "text": "已读取图片",
            "content_blocks": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            ],
        }

    registry = ToolRegistry(
        [
            Tool(
                name="read_file",
                description="read",
                parameters={"type": "object"},
                handler=read_image,
            )
        ]
    )
    call = FunctionCall(id="image-1", name="read_file", arguments={})
    provider = _FakeProvider([_response(calls=(call,)), _response("看到了")])

    await ReActEngine(provider, registry).run((ChatMessage.user("看看图片"),))

    second_messages = provider.calls[1]["messages"]
    assert second_messages[-2].role == "tool"
    assert second_messages[-1].role == "user"
    assert second_messages[-1].content == [
        {"type": "text", "text": "以下是工具 read_file 读取到的文件内容，请直接查看。"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        },
    ]


def _discovery_registry(calls: list[str]) -> ToolRegistry:
    registry = ToolRegistry()

    async def recall() -> str:
        calls.append("recall_memory")
        return "memory"

    async def weather() -> str:
        calls.append("weather_forecast")
        return "sunny"

    registry.register(
        Tool("recall_memory", "memory", {"type": "object"}, recall),
        always_on=True,
    )
    registry.register(
        Tool("weather_forecast", "weather", {"type": "object"}, weather),
        search_hint="天气 预报",
    )
    search = ToolSearchTool(registry)
    registry.register(
        Tool(
            "tool_search",
            "search tools",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            search.execute,
        ),
        always_on=True,
    )
    return registry


async def test_react_tool_filtering_is_disabled_by_default() -> None:
    provider = _FakeProvider([_response("ok")])
    registry = _discovery_registry([])

    await ReActEngine(provider, registry).run((ChatMessage.user("hello"),))

    assert [item["function"]["name"] for item in provider.calls[0]["tools"]] == [
        "recall_memory",
        "weather_forecast",
        "tool_search",
    ]


async def test_react_unlocks_searched_tool_on_the_next_iteration() -> None:
    search_call = FunctionCall(
        id="search", name="tool_search", arguments={"query": "天气"}
    )
    weather_call = FunctionCall(id="weather", name="weather_forecast", arguments={})
    provider = _FakeProvider(
        [
            _response(calls=(search_call,)),
            _response(calls=(weather_call,)),
            _response("sunny"),
        ]
    )
    calls: list[str] = []

    result = await ReActEngine(
        provider,
        _discovery_registry(calls),
        tool_search_enabled=True,
    ).run((ChatMessage.user("weather"),))

    assert [item["function"]["name"] for item in provider.calls[0]["tools"]] == [
        "recall_memory",
        "tool_search",
    ]
    assert [item["function"]["name"] for item in provider.calls[1]["tools"]] == [
        "recall_memory",
        "tool_search",
        "weather_forecast",
    ]
    assert calls == ["weather_forecast"]
    assert result.reply == "sunny"


async def test_react_blocks_direct_call_to_hidden_tool_without_running_handler() -> None:
    hidden_call = FunctionCall(id="hidden", name="weather_forecast", arguments={})
    provider = _FakeProvider([_response(calls=(hidden_call,)), _response("blocked")])
    calls: list[str] = []
    hook_calls: list[str] = []
    registry = _discovery_registry(calls)

    async def audit_hidden(tool_name: str, arguments: dict[str, object]):
        del arguments
        hook_calls.append(tool_name)
        return ToolHookDecision()

    registry.register_hook(ToolHook("audit-hidden", before=audit_hidden))

    result = await ReActEngine(
        provider,
        registry,
        tool_search_enabled=True,
    ).run((ChatMessage.user("weather"),))

    assert calls == []
    assert hook_calls == ["weather_forecast"]
    assert result.tool_chain[0].observation.error_type == "tool_not_loaded"
    assert 'tool_search(query="select:weather_forecast")' in (
        result.tool_chain[0].observation.error_message or ""
    )


async def test_react_reports_stream_and_tool_progress_in_execution_order() -> None:
    call = FunctionCall(id="c1", name="echo", arguments={"text": "hello"})
    provider = _FakeProvider(
        [
            ModelResponse(
                content=None,
                tool_calls=(call,),
                thinking="先调用工具",
                provider_fields={"reasoning_content": "先调用工具"},
            ),
            _response("工具调用完成"),
        ]
    )
    events: list[tuple[str, object]] = []

    class _Progress:
        async def on_stream_delta(self, delta: StreamDelta) -> None:
            events.append(("delta", delta))

        async def on_tool_call_started(self, iteration: int, item: FunctionCall) -> None:
            events.append(("started", (iteration, item.id)))

        async def on_tool_call_completed(
            self,
            iteration: int,
            item: FunctionCall,
            observation: object,
        ) -> None:
            events.append(("completed", (iteration, item.id)))

    result = await ReActEngine(provider, _registry()).run(
        (ChatMessage.user("回显 hello"),),
        progress=_Progress(),
    )

    assert result.reply == "工具调用完成"
    assert events == [
        ("delta", StreamDelta(thinking_delta="先调用工具")),
        ("started", (1, "c1")),
        ("completed", (1, "c1")),
        ("delta", StreamDelta(content_delta="工具调用完成")),
    ]


@pytest.mark.parametrize("failure_point", ["stream", "started", "completed"])
async def test_progress_observer_failure_never_changes_react_result(
    failure_point: str,
) -> None:
    call = FunctionCall(id="c1", name="echo", arguments={"text": "hello"})
    provider = _FakeProvider(
        [
            ModelResponse(
                content=None,
                tool_calls=(call,),
                thinking="先调用工具",
            ),
            _response("工具调用完成"),
        ]
    )

    class _BrokenProgress:
        async def on_stream_delta(self, delta: StreamDelta) -> None:
            if failure_point == "stream":
                raise RuntimeError("stream preview failed")

        async def on_tool_call_started(self, iteration: int, item: FunctionCall) -> None:
            if failure_point == "started":
                raise RuntimeError("tool start preview failed")

        async def on_tool_call_completed(
            self,
            iteration: int,
            item: FunctionCall,
            observation: object,
        ) -> None:
            if failure_point == "completed":
                raise RuntimeError("tool completion preview failed")

    result = await ReActEngine(provider, _registry()).run(
        (ChatMessage.user("回显 hello"),),
        progress=_BrokenProgress(),
    )

    assert result.reply == "工具调用完成"
    assert result.exit_reason == "completed"
    assert result.infrastructure_error is None
    assert result.tool_chain[0].observation.ok is True


async def test_react_replays_provider_fields_with_assistant_tool_call() -> None:
    call = FunctionCall(id="c1", name="echo", arguments={"text": "hello"})
    first = ModelResponse(
        content="",
        tool_calls=(call,),
        finish_reason="tool_calls",
        thinking="先查资料",
        provider_fields={"reasoning_content": "先查资料"},
    )
    provider = _FakeProvider([first, _response("完成")])

    await ReActEngine(provider, _registry()).run((ChatMessage.user("执行"),))

    assistant = provider.calls[1]["messages"][-2]
    assert assistant.role == "assistant"
    assert assistant.provider_fields == {"reasoning_content": "先查资料"}


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


@pytest.mark.parametrize(
    ("denied", "handler", "expected_status"),
    [
        (True, _echo, "denied"),
        (False, _fail, "error"),
    ],
)
async def test_react_emits_distinct_denied_and_error_tool_statuses(
    denied: bool,
    handler: Any,
    expected_status: str,
) -> None:
    async def guard(
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolHookDecision:
        del tool_name, arguments
        if denied:
            return ToolHookDecision(denied=True, reason="blocked")
        return ToolHookDecision()

    call = FunctionCall(id="c1", name="echo", arguments={"text": "x"})
    provider = _FakeProvider([_response(calls=(call,)), _response("已处理")])
    bus = EventBus()
    statuses: list[str] = []
    bus.on(
        "after_tool_result",
        lambda event: statuses.append(event["status"]),
        observer=True,
    )
    registry = _registry(handler)
    registry.register_hook(ToolHook("guard", before=guard))

    result = await ReActEngine(provider, registry, event_bus=bus).run(
        (ChatMessage.user("执行"),)
    )

    assert result.reply == "已处理"
    assert result.tool_chain[0].observation.status == expected_status
    assert statuses == [expected_status]
    tool_message = provider.calls[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert '"ok": false' in (tool_message.content or "")
    await bus.aclose()


async def test_react_forces_tool_free_natural_language_summary_at_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call = FunctionCall(id="c1", name="echo", arguments={"text": "hello"})
    provider = _FakeProvider(
        [
            _response(calls=(call,)),
            _response("目前已取得工具结果，任务先在这里收尾。"),
        ]
    )

    with caplog.at_level(logging.INFO, logger="memopilot.runtime.react"):
        result = await ReActEngine(provider, _registry(), max_iterations=1).run(
            (ChatMessage.user("执行复杂任务"),)
        )

    assert result.reply.startswith("目前")
    assert result.exit_reason == "max_iterations"
    assert len(provider.calls) == 2
    assert provider.calls[-1]["tools"] == ()
    assert provider.calls[-1]["messages"][-1].role == "system"
    assert sum("react_context:" in record.getMessage() for record in caplog.records) == 1


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


async def test_provider_failure_returns_explicit_degraded_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="memopilot.runtime.react"):
        result = await ReActEngine(_FailingProvider(), _registry()).run(
            (ChatMessage.user("hello"),)
        )

    assert result.exit_reason == "provider_error"
    assert result.infrastructure_error == "TimeoutError"
    assert result.reply
    assert any(
        "[llm.error]" in record.getMessage() and "TimeoutError" in record.getMessage()
        for record in caplog.records
    )
    assert sum("react_context:" in record.getMessage() for record in caplog.records) == 1
