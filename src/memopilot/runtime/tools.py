"""统一工具注册、Schema 校验与 Observation 生成。"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Any

from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from memopilot.extensions.hooks import (
    HookContext,
    HookEvent,
    HookTraceItem,
    ToolExecStatus,
    ToolExecutionRequest,
    ToolHook,
)
from memopilot.runtime.contracts import FunctionCall, ToolSchema
from memopilot.runtime.tool_search import (
    META_TOOL_NAMES,
    KeywordSearchBackend,
    ToolDocument,
    ToolMeta,
)

ToolHandler = Callable[..., Any]


def _source_parts(source: str) -> tuple[str, str]:
    if source.startswith("mcp:"):
        return "mcp", source.partition(":")[2].partition("/")[0]
    if source.startswith("plugin:"):
        return "plugin", source.partition(":")[2]
    return "builtin", ""


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
    hook_trace: tuple[HookTraceItem, ...] = ()
    extra_messages: tuple[str, ...] = ()
    retryable: bool = False
    status: ToolExecStatus = "success"

    def __post_init__(self) -> None:
        """保留 ``ok`` 构造兼容，同时把三态状态固化到 Observation。"""
        if self.ok:
            if self.status != "success":
                raise ValueError("成功的工具 Observation 状态必须是 success")
            return
        if self.status == "success":
            derived: ToolExecStatus = "denied" if self.error_type == "hook_denied" else "error"
            object.__setattr__(self, "status", derived)

    @property
    def content(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok, "status": self.status}
        if self.ok:
            if isinstance(self.result, dict) and isinstance(
                self.result.get("content_blocks"), list
            ):
                payload["result"] = self.result.get("text") or "工具执行完成。"
            else:
                payload["result"] = self.result
        else:
            payload["error"] = {
                "type": self.error_type,
                "message": self.error_message,
                "retryable": self.retryable,
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool] = (),
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._metadata: dict[str, ToolMeta] = {}
        self._documents: dict[str, ToolDocument] = {}
        self._search_backend = KeywordSearchBackend()
        for tool in tools:
            self.register(tool)

    def register(
        self,
        tool: Tool,
        *,
        risk: str = "read-only",
        always_on: bool = False,
        search_hint: str | None = None,
        source_type: str | None = None,
        source_name: str = "",
    ) -> None:
        if not tool.name:
            raise ValueError("工具名称不能为空")
        if tool.name in self._tools:
            raise ValueError(f"工具名称重复: {tool.name}")
        if tool.timeout_seconds <= 0:
            raise ValueError(f"工具 {tool.name} 的 timeout_seconds 必须大于 0")
        if not _is_async_callable(tool.handler):
            raise ValueError(
                f"工具 {tool.name} 必须提供异步 Handler；阻塞操作应使用可取消的异步实现或子进程"
            )
        validator_class = validator_for(tool.parameters)
        try:
            validator_class.check_schema(tool.parameters)
        except SchemaError as exc:
            raise ValueError(f"工具 {tool.name} 的 JSON Schema 无效: {exc.message}") from exc
        self._tools[tool.name] = tool
        inferred_type, inferred_name = _source_parts(tool.source)
        meta = ToolMeta(risk=risk, always_on=always_on, search_hint=search_hint)
        document = ToolDocument.from_tool_and_meta(
            tool,
            meta,
            source_type=source_type or inferred_type,
            source_name=source_name or inferred_name,
        )
        self._metadata[tool.name] = meta
        self._documents[tool.name] = document
        self._search_backend.add(document)

    def register_many(
        self,
        tools: Iterable[Tool],
        *,
        risk: str = "read-only",
        source_type: str | None = None,
        source_name: str = "",
    ) -> None:
        previous = dict(self._tools)
        previous_metadata = dict(self._metadata)
        previous_documents = dict(self._documents)
        try:
            for tool in tools:
                self.register(
                    tool,
                    risk=risk,
                    source_type=source_type,
                    source_name=source_name,
                )
        except Exception:
            self._tools = previous
            self._metadata = previous_metadata
            self._documents = previous_documents
            self._search_backend.rebuild(previous_documents.values())
            raise

    def register_existing(self, other: ToolRegistry, names: Iterable[str]) -> None:
        """把共享注册表中的工具复制到临时注册表，不改变共享注册表。"""
        for name in names:
            tool = other.get_tool(name)
            document = other.get_document(name)
            if tool is None or document is None:
                continue
            self.register(
                tool,
                risk=document.risk,
                always_on=document.always_on,
                search_hint=document.search_hint,
                source_type=document.source_type,
                source_name=document.source_name,
            )

    def unregister(self, tool_name: str) -> None:
        self._tools.pop(tool_name, None)
        self._metadata.pop(tool_name, None)
        self._documents.pop(tool_name, None)
        self._search_backend.remove(tool_name)

    def schemas(
        self,
        names: AbstractSet[str] | Iterable[str] | None = None,
    ) -> tuple[ToolSchema, ...]:
        if names is None:
            tools = tuple(self._tools.values())
        elif isinstance(names, AbstractSet):
            tools = tuple(tool for name, tool in self._tools.items() if name in names)
        else:
            tools = tuple(self._tools[name] for name in names if name in self._tools)
        return tuple(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        )

    def get_registered_order(self, names: AbstractSet[str] | None = None) -> list[str]:
        if names is None:
            return list(self._tools)
        return [name for name in self._tools if name in names]

    def get_always_on_names(self) -> set[str]:
        return {name for name, meta in self._metadata.items() if meta.always_on}

    def get_document(self, name: str) -> ToolDocument | None:
        return self._documents.get(name)

    def get_document_results(self, names: Iterable[str]) -> list[dict[str, Any]]:
        return [
            {
                "name": document.name,
                "summary": document.description[:120],
                "why_matched": ["名称:精确匹配"],
                "risk": document.risk,
                "always_on": document.always_on,
            }
            for name in names
            if (document := self._documents.get(name)) is not None
        ]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        excluded_names: AbstractSet[str] | None = None,
    ) -> list[dict[str, Any]]:
        excluded = META_TOOL_NAMES | (set(excluded_names) if excluded_names else set())
        return self._search_backend.search(
            query,
            top_k=top_k,
            allowed_risk=allowed_risk,
            excluded_names=excluded,
        )

    def get_deferred_names(self, visible: set[str] | None = None) -> dict[str, object]:
        excluded = self.get_always_on_names() | META_TOOL_NAMES | (visible or set())
        builtin: list[str] = []
        mcp: dict[str, list[str]] = {}
        for name, document in self._documents.items():
            if name in excluded:
                continue
            if document.source_type == "mcp":
                mcp.setdefault(document.source_name, []).append(name)
            else:
                builtin.append(name)
        return {
            "builtin": sorted(builtin),
            "mcp": {key: sorted(value) for key, value in sorted(mcp.items())},
        }

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def execute(
        self,
        call: FunctionCall,
        *,
        request: ToolExecutionRequest | None = None,
        blocked_reason: str | None = None,
        blocked_error_type: str = "tool_not_loaded",
    ) -> ToolObservation:
        """执行单个工具；Hook 编排由 ToolExecutor 独占。"""
        del request
        tool = self._tools.get(call.name)
        if tool is None:
            return self._failure(call, "unknown_tool", f"工具不存在: {call.name}")
        if call.argument_error is not None:
            return self._failure(call, "invalid_arguments", call.argument_error)
        arguments = dict(call.arguments)
        if blocked_reason is not None:
            return self._failure(
                call,
                blocked_error_type,
                blocked_reason,
                original_arguments=arguments,
                final_arguments=arguments,
                retryable=True,
            )
        try:
            validator_for(tool.parameters)(tool.parameters).validate(arguments)
        except ValidationError as exc:
            return self._failure(
                call,
                "invalid_arguments",
                exc.message,
                original_arguments=arguments,
                final_arguments=arguments,
            )
        try:
            async with asyncio.timeout(tool.timeout_seconds):
                try:
                    result = await tool.handler(**arguments)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return self._failure(
                        call,
                        str(getattr(exc, "error_type", "tool_execution_error")),
                        f"{type(exc).__name__}: {exc}",
                        original_arguments=arguments,
                        final_arguments=arguments,
                        retryable=bool(getattr(exc, "retryable", False)),
                    )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._failure(
                call,
                "tool_timeout",
                f"工具执行超过 {tool.timeout_seconds:g} 秒",
                original_arguments=arguments,
                final_arguments=arguments,
            )
        return ToolObservation(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            result=result,
            original_arguments=arguments,
            final_arguments=arguments,
        )

    @staticmethod
    def _failure(
        call: FunctionCall,
        error_type: str,
        message: str,
        *,
        original_arguments: dict[str, Any] | None = None,
        final_arguments: dict[str, Any] | None = None,
        hook_trace: tuple[HookTraceItem, ...] = (),
        extra_messages: tuple[str, ...] = (),
        retryable: bool = False,
        status: ToolExecStatus = "error",
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
            extra_messages=extra_messages,
            retryable=retryable,
            status=status,
        )


class ToolExecutor:
    """在统一入口编排 ToolHook，Registry 只提供工具查找和低层调用。"""

    def __init__(self, registry: ToolRegistry, hooks: Iterable[ToolHook] = ()) -> None:
        self._registry = registry
        self._hooks: list[ToolHook] = []
        self.register_hooks(hooks)

    def register_hook(self, hook: ToolHook) -> None:
        if any(item.hook_id == hook.hook_id for item in self._hooks):
            raise ValueError(f"ToolHook 重复: {hook.hook_id}")
        self._hooks.append(hook)

    def register_hooks(self, hooks: Iterable[ToolHook]) -> None:
        for hook in hooks:
            self.register_hook(hook)

    def unregister_hook(self, hook_id: str) -> None:
        self._hooks = [hook for hook in self._hooks if hook.hook_id != hook_id]

    def for_registry(self, registry: ToolRegistry) -> ToolExecutor:
        """为临时工具集派生执行器，同时保留同一组 Hook。"""
        if registry is self._registry:
            return self
        return ToolExecutor(registry, tuple(self._hooks))

    async def execute(
        self,
        call: FunctionCall,
        *,
        request: ToolExecutionRequest | None = None,
        blocked_reason: str | None = None,
        blocked_error_type: str = "tool_not_loaded",
    ) -> ToolObservation:
        request = request or ToolExecutionRequest(call.id, call.name, dict(call.arguments))
        original = dict(call.arguments)
        arguments = dict(call.arguments)
        trace: list[HookTraceItem] = []
        messages: list[str] = []
        for hook in self._hooks:
            if hook.event != "pre_tool_use":
                continue
            context = HookContext("pre_tool_use", request, dict(arguments))
            try:
                matched = hook.matches(context)
                if not matched:
                    trace.append(HookTraceItem(hook.hook_id, "pre_tool_use", False))
                    continue
                outcome = await hook.run(context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                trace.append(
                    HookTraceItem(hook.hook_id, "pre_tool_use", True, reason=f"hook failed: {exc}")
                )
                return self._observation(
                    call,
                    "hook_error",
                    f"工具前置 Hook 异常: {hook.hook_id}",
                    original,
                    arguments,
                    trace,
                    messages,
                )
            trace.append(
                HookTraceItem(
                    hook.hook_id,
                    "pre_tool_use",
                    True,
                    outcome.decision,
                    outcome.reason,
                    outcome.extra_message,
                )
            )
            if outcome.extra_message:
                messages.append(outcome.extra_message)
            if outcome.decision == "deny":
                return self._observation(
                    call,
                    "hook_denied",
                    outcome.reason or "Hook 已拒绝工具调用",
                    original,
                    arguments,
                    trace,
                    messages,
                    status="denied",
                )
            if outcome.updated_input is not None:
                arguments = dict(outcome.updated_input)
        observation = await self._registry.execute(
            FunctionCall(call.id, call.name, arguments, call.argument_error),
            request=request,
            blocked_reason=blocked_reason,
            blocked_error_type=blocked_error_type,
        )
        observation = replace(
            observation,
            original_arguments=original,
            final_arguments=arguments,
            hook_trace=tuple(trace),
            extra_messages=tuple(messages),
        )
        return await self._run_post_hooks(request, observation)

    async def _run_post_hooks(
        self, request: ToolExecutionRequest, observation: ToolObservation
    ) -> ToolObservation:
        event: HookEvent = "post_tool_use" if observation.ok else "post_tool_error"
        trace = list(observation.hook_trace)
        messages = list(observation.extra_messages)
        for hook in self._hooks:
            if hook.event != event:
                continue
            context = HookContext(
                event,
                request,
                dict(observation.final_arguments or {}),
                observation.result,
                observation.error_message or "",
            )
            try:
                matched = hook.matches(context)
                if not matched:
                    trace.append(HookTraceItem(hook.hook_id, event, False))
                    continue
                outcome = await hook.run(context)
                if outcome.extra_message:
                    messages.append(outcome.extra_message)
                trace.append(
                    HookTraceItem(
                        hook.hook_id,
                        event,
                        True,
                        outcome.decision,
                        outcome.reason,
                        outcome.extra_message,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                trace.append(HookTraceItem(hook.hook_id, event, True, reason=f"hook failed: {exc}"))
        return replace(observation, hook_trace=tuple(trace), extra_messages=tuple(messages))

    @staticmethod
    def _observation(
        call: FunctionCall,
        error_type: str,
        error_message: str,
        original: dict[str, Any],
        arguments: dict[str, Any],
        trace: list[HookTraceItem],
        messages: list[str],
        *,
        status: ToolExecStatus = "error",
    ) -> ToolObservation:
        return ToolObservation(
            call.id,
            call.name,
            False,
            error_type=error_type,
            error_message=error_message,
            original_arguments=original,
            final_arguments=arguments,
            hook_trace=tuple(trace),
            extra_messages=tuple(messages),
            status=status,
        )


__all__ = ["Tool", "ToolExecutor", "ToolHandler", "ToolObservation", "ToolRegistry"]
