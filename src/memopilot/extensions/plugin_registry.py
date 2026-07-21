"""插件类与装饰器元数据注册表。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MetadataKind(StrEnum):
    LIFECYCLE = "lifecycle"
    TOOL = "tool"
    TOOL_HOOK = "tool_hook"


class PluginEventType(StrEnum):
    BEFORE_TURN = "before_turn"
    BEFORE_REASONING = "before_reasoning"
    PROMPT_RENDER = "prompt_render"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    AFTER_REASONING = "after_reasoning"
    AFTER_TURN = "after_turn"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_RESULT = "after_tool_result"
    PRE_TOOL = "pre_tool"


@dataclass(slots=True)
class PluginHandlerMetadata:
    kind: MetadataKind
    handler: Callable[..., Any]
    handler_name: str
    plugin_module_path: str
    event_type: PluginEventType | None = None
    observer: bool = False
    priority: int = 0
    tool_name: str | None = None
    tool_schema: dict[str, Any] | None = None
    tool_risk: str = "read-write"
    tool_always_on: bool = False
    tool_search_hint: str | None = None
    hook_tool_name: str | None = None


class PluginRegistry:
    def __init__(self) -> None:
        self._handlers: list[PluginHandlerMetadata] = []
        self._classes: dict[str, type[Any]] = {}
        self._instances: dict[str, object] = {}

    def register_class(self, cls: type[Any]) -> None:
        self._classes[cls.__module__] = cls

    def get_class(self, module_path: str) -> type[Any] | None:
        return self._classes.get(module_path)

    def append_handler(self, metadata: PluginHandlerMetadata) -> None:
        duplicate = any(
            item.kind is metadata.kind
            and item.handler_name == metadata.handler_name
            and item.plugin_module_path == metadata.plugin_module_path
            for item in self._handlers
        )
        if not duplicate:
            self._handlers.append(metadata)
            self._handlers.sort(key=lambda item: -item.priority)

    def get_handlers_by_module_path(
        self,
        module_path: str,
    ) -> list[PluginHandlerMetadata]:
        return [item for item in self._handlers if item.plugin_module_path == module_path]

    def register_instance(self, module_path: str, instance: object) -> None:
        self._instances[module_path] = instance

    def get_instance(self, module_path: str) -> object | None:
        return self._instances.get(module_path)

    def remove_plugin(self, module_path: str) -> None:
        self._handlers = [
            item for item in self._handlers if item.plugin_module_path != module_path
        ]
        self._classes.pop(module_path, None)
        self._instances.pop(module_path, None)


plugin_registry = PluginRegistry()


__all__ = [
    "MetadataKind",
    "PluginEventType",
    "PluginHandlerMetadata",
    "PluginRegistry",
    "plugin_registry",
]
