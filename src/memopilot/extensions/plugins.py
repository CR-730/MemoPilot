"""插件加载诊断。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PluginDiagnostic:
    plugin_id: str
    code: str
    message: str

__all__ = ["PluginDiagnostic"]
