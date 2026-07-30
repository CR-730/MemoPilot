from __future__ import annotations

import asyncio
from typing import Any

import pytest

from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolExecutor, ToolRegistry


async def _echo(*, text: str) -> dict[str, str]:
    return {"echo": text}


async def _explode(*, text: str) -> str:
    raise RuntimeError(f"boom:{text}")


def _tool(name: str = "echo", handler: Any = _echo) -> Tool:
    return Tool(
        name=name,
        description="回显文本",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def test_registry_rejects_duplicate_names_and_exposes_function_schema() -> None:
    registry = ToolRegistry([_tool()])

    with pytest.raises(ValueError, match="echo"):
        registry.register(_tool())

    assert registry.schemas() == (
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "回显文本",
                "parameters": _tool().parameters,
            },
        },
    )


def test_registry_rejects_sync_handler() -> None:
    def sync_handler(*, text: str) -> str:
        return text

    with pytest.raises(ValueError, match="异步"):
        ToolRegistry([_tool(handler=sync_handler)])


def test_registry_accepts_async_callable_object() -> None:
    class AsyncHandler:
        async def __call__(self, *, text: str) -> str:
            return text

    ToolRegistry([_tool(handler=AsyncHandler())])


def test_register_many_rolls_back_tools_metadata_and_index_together() -> None:
    registry = ToolRegistry([_tool("existing")])

    with pytest.raises(ValueError, match="existing"):
        registry.register_many([_tool("new"), _tool("existing")])

    assert registry.tool_names == ("existing",)
    assert registry.search("new") == []


async def test_tool_timeout_becomes_react_observation() -> None:
    async def never_returns(*, text: str) -> str:
        await asyncio.Event().wait()
        return text

    tool = _tool(handler=never_returns)
    tool = Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        handler=tool.handler,
        timeout_seconds=0.01,
    )

    observation = await ToolRegistry([tool]).execute(
        FunctionCall(id="1", name="echo", arguments={"text": "x"})
    )

    assert observation.ok is False
    assert observation.error_type == "tool_timeout"
    assert "0.01" in observation.content


async def test_handler_timeout_error_keeps_original_error_semantics() -> None:
    async def raises_timeout(*, text: str) -> str:
        raise TimeoutError("upstream timed out")

    observation = await ToolRegistry([_tool(handler=raises_timeout)]).execute(
        FunctionCall(id="1", name="echo", arguments={"text": "x"})
    )

    assert observation.ok is False
    assert observation.error_type == "tool_execution_error"
    assert observation.error_message == "TimeoutError: upstream timed out"


async def test_external_cancellation_is_not_converted_to_tool_error() -> None:
    entered = asyncio.Event()

    async def cancellable(*, text: str) -> str:
        entered.set()
        await asyncio.Event().wait()
        return text

    execution = asyncio.create_task(
        ToolRegistry([_tool(handler=cancellable)]).execute(
            FunctionCall(id="1", name="echo", arguments={"text": "x"})
        )
    )
    await entered.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution


@pytest.mark.parametrize(
    ("call", "error_type"),
    [
        (FunctionCall(id="1", name="missing", arguments={}), "unknown_tool"),
        (FunctionCall(id="2", name="echo", arguments={}), "invalid_arguments"),
        (
            FunctionCall(
                id="3",
                name="echo",
                arguments={},
                argument_error="invalid JSON",
            ),
            "invalid_arguments",
        ),
    ],
)
async def test_tool_contract_failures_become_observations(
    call: FunctionCall,
    error_type: str,
) -> None:
    observation = await ToolRegistry([_tool()]).execute(call)

    assert observation.ok is False
    assert observation.error_type == error_type
    assert observation.call_id == call.id


async def test_tool_execution_exception_becomes_observation() -> None:
    observation = await ToolRegistry([_tool(handler=_explode)]).execute(
        FunctionCall(id="1", name="echo", arguments={"text": "x"})
    )

    assert observation.ok is False
    assert observation.error_type == "tool_execution_error"
    assert "RuntimeError" in observation.content


async def test_successful_tool_result_is_serializable_observation() -> None:
    observation = await ToolRegistry([_tool()]).execute(
        FunctionCall(id="1", name="echo", arguments={"text": "hello"})
    )

    assert observation.ok is True
    assert observation.result == {"echo": "hello"}
    assert '"ok": true' in observation.content
    assert not hasattr(observation, "side_effect_status")
    assert not hasattr(_tool(), "side_effect_class")


async def test_executor_owns_tool_invocation() -> None:
    observation = await ToolExecutor(ToolRegistry([_tool()])).execute(
        FunctionCall(id="1", name="echo", arguments={"text": "hello"})
    )

    assert observation.ok is True
    assert observation.result == {"echo": "hello"}
