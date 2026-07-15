"""统一工具注册、Schema 校验与 Observation 生成。"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from memopilot.extensions.hooks import ToolHook
from memopilot.runtime.contracts import FunctionCall, ToolSchema

ToolHandler = Callable[..., Any]


def _is_async_callable(handler: ToolHandler) -> bool:
    candidates = (handler, type(handler).__call__)
    for candidate in candidates:
        try:
            candidate = inspect.unwrap(candidate)
        except ValueError:
            pass
        if inspect.iscoroutinefunction(candidate):
            return True
    return False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    timeout_seconds: float = 30.0
    source: str = "builtin"
    side_effect_class: str = "none"


@dataclass(frozen=True)
class ToolObservation:
    call_id: str
    tool_name: str
    ok: bool
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None
    original_arguments: dict[str, Any] | None = None
    final_arguments: dict[str, Any] | None = None
    hook_trace: tuple[str, ...] = ()
    retryable: bool = False
    side_effect_status: str = "not_started"

    @property
    def content(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["result"] = self.result
        else:
            payload["error"] = {
                "type": self.error_type,
                "message": self.error_message,
                "retryable": self.retryable,
                "side_effect_status": self.side_effect_status,
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        hooks: Iterable[ToolHook] = (),
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._hooks: list[ToolHook] = []
        for tool in tools:
            self.register(tool)
        for hook in hooks:
            self.register_hook(hook)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("工具名称不能为空")
        if tool.name in self._tools:
            raise ValueError(f"工具名称重复: {tool.name}")
        if tool.timeout_seconds <= 0:
            raise ValueError(f"工具 {tool.name} 的 timeout_seconds 必须大于 0")
        if tool.side_effect_class not in {"none", "idempotent", "non_idempotent"}:
            raise ValueError(f"工具 {tool.name} 的 side_effect_class 无效")
        if not _is_async_callable(tool.handler):
            raise ValueError(
                f"工具 {tool.name} 必须提供异步 Handler；"
                "阻塞操作应使用可取消的异步实现或子进程"
            )
        validator_class = validator_for(tool.parameters)
        try:
            validator_class.check_schema(tool.parameters)
        except SchemaError as exc:
            raise ValueError(f"工具 {tool.name} 的 JSON Schema 无效: {exc.message}") from exc
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        previous = dict(self._tools)
        try:
            for tool in tools:
                self.register(tool)
        except Exception:
            self._tools = previous
            raise

    def register_hook(self, hook: ToolHook) -> None:
        if any(existing.hook_id == hook.hook_id for existing in self._hooks):
            raise ValueError(f"ToolHook 重复: {hook.hook_id}")
        self._hooks.append(hook)

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

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    async def execute(self, call: FunctionCall) -> ToolObservation:
        tool = self._tools.get(call.name)
        if tool is None:
            return await self._run_after_hooks(
                call.name,
                self._failure(call, "unknown_tool", f"工具不存在: {call.name}"),
            )
        if call.argument_error is not None:
            return await self._run_after_hooks(
                call.name,
                self._failure(call, "invalid_arguments", call.argument_error),
            )
        original_arguments = dict(call.arguments)
        final_arguments = dict(call.arguments)
        hook_trace: list[str] = []
        for hook in self._hooks:
            if hook.before is None:
                continue
            try:
                decision = await hook.before(call.name, dict(final_arguments))
            except asyncio.CancelledError:
                raise
            except Exception:
                hook_trace.append(f"{hook.hook_id}:error")
                return await self._run_after_hooks(
                    call.name,
                    self._failure(
                        call,
                        "hook_error",
                        f"工具前置 Hook 异常: {hook.hook_id}",
                        original_arguments=original_arguments,
                        final_arguments=final_arguments,
                        hook_trace=tuple(hook_trace),
                    ),
                )
            if decision is None:
                hook_trace.append(f"{hook.hook_id}:passed")
                continue
            if decision.denied:
                hook_trace.append(f"{hook.hook_id}:denied")
                return await self._run_after_hooks(
                    call.name,
                    self._failure(
                        call,
                        "hook_denied",
                        decision.reason or "Hook 已拒绝工具调用",
                        original_arguments=original_arguments,
                        final_arguments=final_arguments,
                        hook_trace=tuple(hook_trace),
                    ),
                )
            if decision.arguments is not None:
                final_arguments = dict(decision.arguments)
                hook_trace.append(f"{hook.hook_id}:rewritten")
            else:
                hook_trace.append(f"{hook.hook_id}:passed")
        try:
            validator_for(tool.parameters)(tool.parameters).validate(final_arguments)
        except ValidationError as exc:
            return await self._run_after_hooks(
                call.name,
                self._failure(
                    call,
                    "invalid_arguments",
                    exc.message,
                    original_arguments=original_arguments,
                    final_arguments=final_arguments,
                    hook_trace=tuple(hook_trace),
                ),
            )
        try:
            async with asyncio.timeout(tool.timeout_seconds):
                try:
                    result = await tool.handler(**final_arguments)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return await self._run_after_hooks(
                        call.name,
                        self._failure(
                            call,
                            str(getattr(exc, "error_type", "tool_execution_error")),
                            f"{type(exc).__name__}: {exc}",
                            original_arguments=original_arguments,
                            final_arguments=final_arguments,
                            hook_trace=tuple(hook_trace),
                            retryable=bool(getattr(exc, "retryable", False)),
                            side_effect_status=str(
                                getattr(exc, "side_effect_status", "unknown")
                            ),
                        ),
                    )
        except TimeoutError:
            return await self._run_after_hooks(
                call.name,
                self._failure(
                    call,
                    "tool_timeout",
                    f"工具执行超过 {tool.timeout_seconds:g} 秒",
                    original_arguments=original_arguments,
                    final_arguments=final_arguments,
                    hook_trace=tuple(hook_trace),
                ),
            )
        except asyncio.CancelledError:
            raise
        observation = ToolObservation(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            result=result,
            original_arguments=original_arguments,
            final_arguments=final_arguments,
            hook_trace=tuple(hook_trace),
            side_effect_status=(
                "not_applicable" if tool.side_effect_class == "none" else "confirmed"
            ),
        )
        return await self._run_after_hooks(call.name, observation)

    async def _run_after_hooks(
        self,
        tool_name: str,
        observation: ToolObservation,
    ) -> ToolObservation:
        hook_trace = list(observation.hook_trace)
        for hook in self._hooks:
            if hook.after is None:
                continue
            try:
                await hook.after(tool_name, observation)
                hook_trace.append(f"{hook.hook_id}:observed")
            except asyncio.CancelledError:
                raise
            except Exception:
                hook_trace.append(f"{hook.hook_id}:after_error")
        return replace(observation, hook_trace=tuple(hook_trace))

    @staticmethod
    def _failure(
        call: FunctionCall,
        error_type: str,
        message: str,
        *,
        original_arguments: dict[str, Any] | None = None,
        final_arguments: dict[str, Any] | None = None,
        hook_trace: tuple[str, ...] = (),
        retryable: bool = False,
        side_effect_status: str = "not_started",
    ) -> ToolObservation:
        return ToolObservation(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            error_type=error_type,
            error_message=message,
            original_arguments=original_arguments,
            final_arguments=final_arguments,
            hook_trace=hook_trace,
            retryable=retryable,
            side_effect_status=side_effect_status,
        )


__all__ = ["Tool", "ToolHandler", "ToolObservation", "ToolRegistry"]
