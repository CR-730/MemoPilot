"""原型兼容的插件声明装饰器。"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, get_args, get_origin

from memopilot.extensions.plugin_registry import (
    MetadataKind,
    PluginEventType,
    PluginHandlerMetadata,
    plugin_registry,
)


def _lifecycle_decorator(
    event_type: PluginEventType,
    *,
    observer: bool,
    priority: int,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        plugin_registry.append_handler(
            PluginHandlerMetadata(
                kind=MetadataKind.LIFECYCLE,
                event_type=event_type,
                handler=func,
                handler_name=func.__name__,
                plugin_module_path=func.__module__,
                observer=observer,
                priority=priority,
            )
        )
        return func

    return decorate


def on_before_turn(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.BEFORE_TURN, observer=False, priority=priority
    )


def on_before_reasoning(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.BEFORE_REASONING, observer=False, priority=priority
    )


def on_prompt_render(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.PROMPT_RENDER, observer=False, priority=priority
    )


def on_before_step(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.BEFORE_STEP, observer=False, priority=priority
    )


def on_after_step(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.AFTER_STEP, observer=True, priority=priority
    )


def on_after_reasoning(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.AFTER_REASONING, observer=False, priority=priority
    )


def on_after_turn(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.AFTER_TURN, observer=True, priority=priority
    )


def on_tool_call(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.BEFORE_TOOL_CALL, observer=True, priority=priority
    )


def on_tool_result(
    *, priority: int = 0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lifecycle_decorator(
        PluginEventType.AFTER_TOOL_RESULT, observer=True, priority=priority
    )


def on_tool_pre(
    *,
    tool_name: str | None = None,
    priority: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        plugin_registry.append_handler(
            PluginHandlerMetadata(
                kind=MetadataKind.TOOL_HOOK,
                event_type=PluginEventType.PRE_TOOL,
                handler=func,
                handler_name=func.__name__,
                plugin_module_path=func.__module__,
                hook_tool_name=tool_name,
                priority=priority,
            )
        )
        return func

    return decorate


def tool(
    name: str,
    *,
    risk: str = "read-write",
    always_on: bool = False,
    search_hint: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        params = list(inspect.signature(func).parameters)
        if len(params) < 2 or params[:2] != ["self", "event"]:
            raise TypeError(
                f"@tool handler 前两个参数必须是 self 和 event: {func.__qualname__}"
            )
        plugin_registry.append_handler(
            PluginHandlerMetadata(
                kind=MetadataKind.TOOL,
                handler=func,
                handler_name=func.__name__,
                plugin_module_path=func.__module__,
                tool_name=name,
                tool_schema=_derive_params_schema(func),
                tool_risk=risk,
                tool_always_on=always_on,
                tool_search_hint=search_hint,
            )
        )
        return func

    return decorate


def _derive_params_schema(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    descriptions = {
        name: value.strip()
        for name, value in re.findall(
            r"^\s*:param\s+(\w+)\s*:\s*(.+)$",
            func.__doc__ or "",
            flags=re.MULTILINE,
        )
    }
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "event"}:
            continue
        schema = {"type": _json_type(parameter.annotation)}
        if name in descriptions:
            schema["description"] = descriptions[name]
        properties[name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _json_type(annotation: Any) -> str:
    if isinstance(annotation, str):
        normalized = annotation.strip()
        return {
            "str": "string",
            "int": "number",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
        }.get(normalized, "string")
    origin = get_origin(annotation)
    candidate = origin or annotation
    if origin is not None and type(None) in get_args(annotation):
        candidate = next(item for item in get_args(annotation) if item is not type(None))
    return {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }.get(candidate, "string")


__all__ = [
    "on_after_reasoning",
    "on_after_step",
    "on_after_turn",
    "on_before_reasoning",
    "on_before_step",
    "on_before_turn",
    "on_prompt_render",
    "on_tool_call",
    "on_tool_pre",
    "on_tool_result",
    "tool",
]
