"""可信本地 Python 插件的静态发现与原子注册。"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import yaml

from memopilot.extensions.events import EventHandler
from memopilot.extensions.hooks import ToolHook
from memopilot.extensions.prompts import PromptBlock, PromptRenderer
from memopilot.runtime.phases import PhaseModule
from memopilot.runtime.tools import Tool, ToolRegistry

PLUGIN_API_VERSION = 1
_PLUGIN_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_CAPABILITIES = {
    "tools",
    "prompt_blocks",
    "phase_modules",
    "tool_hooks",
    "event_handlers",
}


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    version: str
    api_version: int
    entrypoint: str
    requires: tuple[str, ...]
    capabilities: frozenset[str]
    directory: Path


@dataclass(frozen=True, slots=True)
class PluginDiagnostic:
    plugin_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExtensionRegistry:
    tools: tuple[Tool, ...] = ()
    prompt_blocks: tuple[PromptBlock, ...] = ()
    phase_modules: tuple[PhaseModule, ...] = ()
    tool_hooks: tuple[ToolHook, ...] = ()
    event_handlers: tuple[EventHandler, ...] = ()

    def merge(self, staged: ExtensionRegistry) -> ExtensionRegistry:
        candidate = ExtensionRegistry(
            self.tools + staged.tools,
            self.prompt_blocks + staged.prompt_blocks,
            self.phase_modules + staged.phase_modules,
            self.tool_hooks + staged.tool_hooks,
            self.event_handlers + staged.event_handlers,
        )
        ToolRegistry(candidate.tools, hooks=candidate.tool_hooks)
        PromptRenderer(candidate.prompt_blocks)
        _ensure_unique(
            (module.slot for module in candidate.phase_modules),
            "PhaseModule slot",
        )
        _ensure_unique(
            (f"{handler.event}/{handler.handler_id}" for handler in candidate.event_handlers),
            "EventHandler",
        )
        return candidate


class PluginContext:
    def __init__(self, manifest: PluginManifest) -> None:
        self._manifest = manifest
        self._tools: list[Tool] = []
        self._prompt_blocks: list[PromptBlock] = []
        self._phase_modules: list[PhaseModule] = []
        self._tool_hooks: list[ToolHook] = []
        self._event_handlers: list[EventHandler] = []

    def register_tool(self, tool: Tool) -> None:
        self._require("tools")
        self._tools.append(tool)

    def register_prompt_block(self, block: PromptBlock) -> None:
        self._require("prompt_blocks")
        self._prompt_blocks.append(block)

    def register_phase_module(self, module: PhaseModule) -> None:
        self._require("phase_modules")
        self._phase_modules.append(module)

    def register_tool_hook(self, hook: ToolHook) -> None:
        self._require("tool_hooks")
        self._tool_hooks.append(hook)

    def register_event_handler(self, handler: EventHandler) -> None:
        self._require("event_handlers")
        self._event_handlers.append(handler)

    def freeze(self) -> ExtensionRegistry:
        return ExtensionRegistry(
            tuple(self._tools),
            tuple(self._prompt_blocks),
            tuple(self._phase_modules),
            tuple(self._tool_hooks),
            tuple(self._event_handlers),
        )

    def _require(self, capability: str) -> None:
        if capability not in self._manifest.capabilities:
            raise ValueError(
                f"插件 {self._manifest.plugin_id} 未声明 capability: {capability}"
            )


@dataclass(frozen=True, slots=True)
class PluginLoadResult:
    registry: ExtensionRegistry
    enabled_plugin_ids: tuple[str, ...]
    diagnostics: tuple[PluginDiagnostic, ...]


class PluginRuntime:
    def load_directory(self, root: Path) -> PluginLoadResult:
        manifests: dict[str, PluginManifest] = {}
        diagnostics: list[PluginDiagnostic] = []
        for path in sorted(root.glob("*/manifest.yaml")):
            fallback_id = path.parent.name
            try:
                manifest = _read_manifest(path)
                if manifest.plugin_id in manifests:
                    raise ValueError(f"plugin_id 重复: {manifest.plugin_id}")
                manifests[manifest.plugin_id] = manifest
            except Exception as exc:
                diagnostics.append(
                    PluginDiagnostic(fallback_id, "invalid_manifest", str(exc))
                )

        order, dependency_diagnostics = _dependency_order(manifests)
        diagnostics.extend(dependency_diagnostics)
        disabled = {item.plugin_id for item in dependency_diagnostics}
        registry = ExtensionRegistry()
        enabled: list[str] = []
        for plugin_id in order:
            manifest = manifests[plugin_id]
            if plugin_id in disabled or any(item not in enabled for item in manifest.requires):
                if plugin_id not in disabled:
                    diagnostics.append(
                        PluginDiagnostic(
                            plugin_id,
                            "disabled_dependency",
                            "依赖插件未成功启用",
                        )
                    )
                continue
            context = PluginContext(manifest)
            try:
                register = _load_entrypoint(manifest)
                register(context)
                registry = registry.merge(context.freeze())
            except Exception as exc:
                diagnostics.append(
                    PluginDiagnostic(plugin_id, "registration_failed", str(exc))
                )
                continue
            enabled.append(plugin_id)
        return PluginLoadResult(registry, tuple(enabled), tuple(diagnostics))


def _read_manifest(path: Path) -> PluginManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest 必须是 YAML 对象")
    required = {
        "plugin_id",
        "version",
        "api_version",
        "entrypoint",
        "requires",
        "capabilities",
    }
    missing = required.difference(raw)
    unknown = set(raw).difference(required)
    if missing:
        raise ValueError("manifest 缺少字段: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError("manifest 包含未知字段: " + ", ".join(sorted(unknown)))
    plugin_id = str(raw["plugin_id"])
    if not _PLUGIN_ID.fullmatch(plugin_id):
        raise ValueError(f"plugin_id 格式无效: {plugin_id}")
    api_version = int(raw["api_version"])
    if api_version != PLUGIN_API_VERSION:
        raise ValueError(f"不支持 plugin api_version={api_version}")
    requires = _string_tuple(raw["requires"], "requires")
    capabilities = frozenset(_string_tuple(raw["capabilities"], "capabilities"))
    unsupported = capabilities.difference(_CAPABILITIES)
    if unsupported:
        raise ValueError("未知 capabilities: " + ", ".join(sorted(unsupported)))
    entrypoint = str(raw["entrypoint"])
    if ":" not in entrypoint:
        raise ValueError("entrypoint 必须为相对文件:函数")
    return PluginManifest(
        plugin_id,
        str(raw["version"]),
        api_version,
        entrypoint,
        requires,
        capabilities,
        path.parent.resolve(),
    )


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"manifest {field} 必须是字符串数组")
    return tuple(value)


def _dependency_order(
    manifests: dict[str, PluginManifest],
) -> tuple[list[str], list[PluginDiagnostic]]:
    diagnostics: list[PluginDiagnostic] = []
    valid: set[str] = set(manifests)
    for manifest in manifests.values():
        missing = set(manifest.requires).difference(manifests)
        if missing:
            valid.discard(manifest.plugin_id)
            diagnostics.append(
                PluginDiagnostic(
                    manifest.plugin_id,
                    "missing_dependency",
                    "缺少依赖: " + ", ".join(sorted(missing)),
                )
            )
    indegree = {plugin_id: 0 for plugin_id in valid}
    dependents: dict[str, list[str]] = {plugin_id: [] for plugin_id in valid}
    for plugin_id in valid:
        for dependency in manifests[plugin_id].requires:
            if dependency not in valid:
                continue
            indegree[plugin_id] += 1
            dependents[dependency].append(plugin_id)
    ready = sorted(plugin_id for plugin_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        plugin_id = ready.pop(0)
        order.append(plugin_id)
        for dependent in dependents[plugin_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    cycles = sorted(plugin_id for plugin_id, degree in indegree.items() if degree > 0)
    diagnostics.extend(
        PluginDiagnostic(plugin_id, "dependency_cycle", "插件依赖存在循环")
        for plugin_id in cycles
    )
    order.extend(sorted(set(manifests).difference(order)))
    return order, diagnostics


def _load_entrypoint(manifest: PluginManifest) -> Callable[[PluginContext], None]:
    relative_file, attribute = manifest.entrypoint.rsplit(":", 1)
    file_path = (manifest.directory / relative_file).resolve()
    try:
        file_path.relative_to(manifest.directory)
    except ValueError as exc:
        raise ValueError("插件 entrypoint 不能越出插件目录") from exc
    if not file_path.is_file():
        raise ValueError(f"插件入口文件不存在: {relative_file}")
    module_name = f"_memopilot_plugin_{manifest.plugin_id.replace('.', '_').replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载插件入口: {relative_file}")
    module = importlib.util.module_from_spec(spec)
    _execute_module(spec.loader.exec_module, module)
    register = getattr(module, attribute, None)
    if not callable(register):
        raise ValueError(f"插件入口不可调用: {attribute}")
    return cast(Callable[[PluginContext], None], register)


def _execute_module(execute: Callable[[ModuleType], None], module: ModuleType) -> None:
    execute(module)


def _ensure_unique(values: Any, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{label} 重复: {value}")
        seen.add(value)


__all__ = [
    "ExtensionRegistry",
    "PLUGIN_API_VERSION",
    "PluginContext",
    "PluginDiagnostic",
    "PluginLoadResult",
    "PluginManifest",
    "PluginRuntime",
]
