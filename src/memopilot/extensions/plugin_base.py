"""原型式 Python 插件基类。"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memopilot.extensions.plugin_context import PluginContext


class Plugin(ABC):
    name: str | None = None
    version: str | None = None
    desc: str | None = None
    author: str | None = None
    context: PluginContext

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        from memopilot.extensions.plugin_registry import plugin_registry

        plugin_registry.register_class(cls)

    async def initialize(self) -> None:
        return None

    async def terminate(self) -> None:
        return None

    def before_turn_modules(self) -> list[object]:
        return []

    def before_reasoning_modules(self) -> list[object]:
        return []

    def prompt_render_modules(self) -> list[object]:
        return []

    def before_step_modules(self) -> list[object]:
        return []

    def after_step_modules(self) -> list[object]:
        return []

    def after_reasoning_modules(self) -> list[object]:
        return []

    def after_turn_modules(self) -> list[object]:
        return []


__all__ = ["Plugin"]
