"""统一工具注册、Schema 校验与 Observation 生成。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from memopilot.runtime.contracts import FunctionCall, ToolSchema

ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


@dataclass(frozen=True)
class ToolObservation:
    call_id: str
    tool_name: str
    ok: bool
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def content(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["result"] = self.result
        else:
            payload["error"] = {
                "type": self.error_type,
                "message": self.error_message,
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("工具名称不能为空")
        if tool.name in self._tools:
            raise ValueError(f"工具名称重复: {tool.name}")
        validator_class = validator_for(tool.parameters)
        try:
            validator_class.check_schema(tool.parameters)
        except SchemaError as exc:
            raise ValueError(f"工具 {tool.name} 的 JSON Schema 无效: {exc.message}") from exc
        self._tools[tool.name] = tool

    def schemas(self) -> tuple[ToolSchema, ...]:
        return tuple(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        )

    async def execute(self, call: FunctionCall) -> ToolObservation:
        tool = self._tools.get(call.name)
        if tool is None:
            return self._failure(call, "unknown_tool", f"工具不存在: {call.name}")
        if call.argument_error is not None:
            return self._failure(call, "invalid_arguments", call.argument_error)
        try:
            validator_for(tool.parameters)(tool.parameters).validate(call.arguments)
        except ValidationError as exc:
            return self._failure(call, "invalid_arguments", exc.message)
        try:
            result = await tool.handler(**call.arguments)
        except Exception as exc:
            return self._failure(
                call,
                "tool_execution_error",
                f"{type(exc).__name__}: {exc}",
            )
        return ToolObservation(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            result=result,
        )

    @staticmethod
    def _failure(
        call: FunctionCall,
        error_type: str,
        message: str,
    ) -> ToolObservation:
        return ToolObservation(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            error_type=error_type,
            error_message=message,
        )


__all__ = ["Tool", "ToolHandler", "ToolObservation", "ToolRegistry"]
