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
    HookOutcome,
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
            derived: ToolExecStatus = (
                "denied" if self.error_type == "hook_denied" else "error"
            )
            object.__setattr__(self, "status", derived)

    @property
    def legacy_hook_trace(self) -> tuple[str, ...]:
        """供仍只展示摘要的旧调用方读取；事实源保持结构化。"""
        return tuple(
            f"{item.hook_name}:{item.event}:"
            f"{'matched' if item.matched else 'unmatched'}:{item.decision}"
            for item in self.hook_trace
        )

    @property
    def pre_hook_trace(self) -> tuple[HookTraceItem, ...]:
        return tuple(item for item in self.hook_trace if item.event == "pre_tool_use")

    @property
    def post_hook_trace(self) -> tuple[HookTraceItem, ...]:
        return tuple(item for item in self.hook_trace if item.event != "pre_tool_use")

    @property
    def content(self) -> str:
        payload: dict[str, Any] = {"ok": self.ok, "status": self.status}
        if self.ok:
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
        *,
        hooks: Iterable[ToolHook] = (),
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._metadata: dict[str, ToolMeta] = {}
        self._documents: dict[str, ToolDocument] = {}
        self._search_backend = KeywordSearchBackend()
        self._hooks: list[ToolHook] = []
        for tool in tools:
            self.register(tool)
        for hook in hooks:
            self.register_hook(hook)

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
                f"工具 {tool.name} 必须提供异步 Handler；"
                "阻塞操作应使用可取消的异步实现或子进程"
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

    def register_hook(self, hook: ToolHook) -> None:
        if any(existing.hook_id == hook.hook_id for existing in self._hooks):
            raise ValueError(f"ToolHook 重复: {hook.hook_id}")
        self._hooks.append(hook)

    def register_hooks(self, hooks: Iterable[ToolHook]) -> None:
        previous = list(self._hooks)
        try:
            for hook in hooks:
                self.register_hook(hook)
        except Exception:
            self._hooks = previous
            raise

    def unregister_hook(self, hook_id: str) -> None:
        self._hooks = [hook for hook in self._hooks if hook.hook_id != hook_id]

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
        execution_request = request or ToolExecutionRequest(
            call_id=call.id,
            tool_name=call.name,
            arguments=dict(call.arguments),
        )
        execution_request.call_id = call.id
        execution_request.tool_name = call.name
        execution_request.arguments = dict(call.arguments)
        tool = self._tools.get(call.name)
        if tool is None:
            return await self._run_after_hooks(
                call.name,
                self._failure(call, "unknown_tool", f"工具不存在: {call.name}"),
                execution_request,
            )
        if call.argument_error is not None:
            return await self._run_after_hooks(
                call.name,
                self._failure(call, "invalid_arguments", call.argument_error),
                execution_request,
            )
        original_arguments = dict(call.arguments)
        final_arguments = dict(call.arguments)
        hook_trace: list[HookTraceItem] = []
        extra_messages: list[str] = []
        for hook in self._hooks:
            if hook.event != "pre_tool_use" and hook.before is None:
                continue
            matched = False
            try:
                if hook.event == "pre_tool_use":
                    context = HookContext(
                        event="pre_tool_use",
                        request=execution_request,
                        current_arguments=dict(final_arguments),
                    )
                    matched = hook.matches(context)
                    if not matched:
                        hook_trace.append(
                            HookTraceItem(
                                hook_name=hook.hook_id,
                                event="pre_tool_use",
                                matched=False,
                            )
                        )
                        continue
                    decision = await hook.run(context)
                else:
                    matched = True
                    decision = await self._run_legacy_pre_hook(
                        hook,
                        execution_request,
                        final_arguments,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                hook_trace.append(
                    HookTraceItem(
                        hook_name=hook.hook_id,
                        event="pre_tool_use",
                        matched=matched,
                        reason=f"hook failed: {exc}",
                    )
                )
                return self._failure(
                    call,
                    "hook_error",
                    f"工具前置 Hook 异常: {hook.hook_id}",
                    original_arguments=original_arguments,
                    final_arguments=final_arguments,
                    hook_trace=tuple(hook_trace),
                    extra_messages=tuple(extra_messages),
                )
            if decision.extra_message:
                extra_messages.append(decision.extra_message)
            hook_trace.append(
                HookTraceItem(
                    hook_name=hook.hook_id,
                    event="pre_tool_use",
                    matched=True,
                    decision=decision.decision,
                    reason=decision.reason,
                    extra_message=decision.extra_message,
                )
            )
            if decision.decision == "deny":
                return self._failure(
                    call,
                    "hook_denied",
                    decision.reason or "Hook 已拒绝工具调用",
                    status="denied",
                    original_arguments=original_arguments,
                    final_arguments=final_arguments,
                    hook_trace=tuple(hook_trace),
                    extra_messages=tuple(extra_messages),
                )
            if decision.updated_input is not None:
                final_arguments = dict(decision.updated_input)
        if blocked_reason is not None:
            return self._failure(
                call,
                blocked_error_type,
                blocked_reason,
                original_arguments=original_arguments,
                final_arguments=final_arguments,
                hook_trace=tuple(hook_trace),
                extra_messages=tuple(extra_messages),
                retryable=True,
            )
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
                execution_request,
                extra_messages=extra_messages,
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
                        ),
                        execution_request,
                        extra_messages=extra_messages,
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
                execution_request,
                extra_messages=extra_messages,
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
            extra_messages=tuple(extra_messages),
            status="success",
        )
        return await self._run_after_hooks(
            call.name,
            observation,
            execution_request,
            extra_messages=extra_messages,
        )

    async def _run_legacy_pre_hook(
        self,
        hook: ToolHook,
        request: ToolExecutionRequest,
        arguments: dict[str, Any],
    ) -> HookOutcome:
        if hook.before is None:
            return HookOutcome()
        decision = await hook.before(request.tool_name, dict(arguments))
        if decision is None:
            return HookOutcome()
        return HookOutcome(
            decision="deny" if decision.denied else "pass",
            updated_input=(
                dict(decision.arguments) if decision.arguments is not None else None
            ),
            reason=decision.reason or "",
        )

    async def _run_after_hooks(
        self,
        tool_name: str,
        observation: ToolObservation,
        request: ToolExecutionRequest,
        *,
        extra_messages: list[str] | None = None,
    ) -> ToolObservation:
        hook_trace: list[HookTraceItem] = list(observation.hook_trace)
        messages = list(observation.extra_messages)
        if extra_messages:
            messages = list(extra_messages)
        for hook in self._hooks:
            expected: HookEvent = (
                "post_tool_use" if observation.status == "success" else "post_tool_error"
            )
            if hook.after is None and hook.event != expected:
                continue
            matched = False
            try:
                if hook.after is not None:
                    matched = True
                    await hook.after(tool_name, observation)
                    outcome = HookOutcome()
                else:
                    context = HookContext(
                        event=expected,
                        request=request,
                        current_arguments=dict(observation.final_arguments or {}),
                        result=observation.result if observation.ok else "",
                        error=(observation.error_message or "") if not observation.ok else "",
                    )
                    matched = hook.matches(context)
                    if not matched:
                        hook_trace.append(
                            HookTraceItem(
                                hook_name=hook.hook_id,
                                event=expected,
                                matched=False,
                            )
                        )
                        continue
                    outcome = await hook.run(context)
                    if outcome.extra_message:
                        messages.append(outcome.extra_message)
                hook_trace.append(
                    HookTraceItem(
                        hook_name=hook.hook_id,
                        event=expected,
                        matched=True,
                        decision=outcome.decision,
                        reason=outcome.reason,
                        extra_message=outcome.extra_message,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                hook_trace.append(
                    HookTraceItem(
                        hook_name=hook.hook_id,
                        event=expected,
                        matched=matched,
                        reason=f"hook failed: {exc}",
                    )
                )
        return replace(
            observation,
            hook_trace=tuple(hook_trace),
            extra_messages=tuple(messages),
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


__all__ = ["Tool", "ToolHandler", "ToolObservation", "ToolRegistry"]
