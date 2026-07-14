from __future__ import annotations

from typing import Any

import pytest

from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import Tool, ToolRegistry


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

