"""基于原型合同的本地 Python 插件管理器。"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from memopilot.extensions.events import EventHandler
from memopilot.extensions.hooks import (
    HookContext,
    HookOutcome,
    ToolExecutionRequest,
    ToolHook,
    ToolHookDecision,
)
from memopilot.extensions.plugin_config import PluginConfig
from memopilot.extensions.plugin_context import PluginContext, PluginKVStore
from memopilot.extensions.plugin_events import PluginEventContext, PreToolCtx
from memopilot.extensions.plugin_registry import (
    MetadataKind,
    PluginHandlerMetadata,
    plugin_registry,
)
from memopilot.extensions.plugins import PluginDiagnostic
from memopilot.extensions.prompts import PromptBlock
from memopilot.runtime.phases import (
    LifecyclePhase,
    PhaseContext,
    PhaseModule,
    PluginPhaseFrame,
)
from memopilot.runtime.tools import Tool, ToolRegistry

_MODULE_METHODS = (
    "before_turn_modules",
    "before_reasoning_modules",
    "prompt_render_modules",
    "before_step_modules",
    "after_step_modules",
    "after_reasoning_modules",
    "after_turn_modules",
)

_METHOD_PHASES = {
    "before_turn_modules": LifecyclePhase.BEFORE_TURN,
    "before_reasoning_modules": LifecyclePhase.BEFORE_REASONING,
    "prompt_render_modules": LifecyclePhase.PROMPT_RENDER,
    "before_step_modules": LifecyclePhase.BEFORE_STEP,
    "after_step_modules": LifecyclePhase.AFTER_STEP,
    "after_reasoning_modules": LifecyclePhase.AFTER_REASONING,
    "after_turn_modules": LifecyclePhase.AFTER_TURN,
}


def _plugin_state_root(workspace: Path | None) -> Path:
    if workspace is None:
        return Path(tempfile.mkdtemp(prefix="memopilot-plugin-state-"))
    return workspace.expanduser().resolve() / ".memopilot" / "plugin_state"


def _safe_plugin_state_name(plugin_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", plugin_id).strip("._-")
    if normalized and normalized == plugin_id and plugin_id not in {".", ".."}:
        return normalized
    digest = hashlib.sha256(plugin_id.encode("utf-8")).hexdigest()[:12]
    return f"{normalized or 'plugin'}-{digest}"


@dataclass(frozen=True, slots=True)
class _PhaseAdapter:
    phase: LifecyclePhase
    delegate: Any
    slot: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    declared_produces: tuple[str, ...]

    @property
    def optional_produces(self) -> tuple[str, ...]:
        """PhaseFrame 插件按原型可条件性写入声明的 slot。"""
        return self.declared_produces

    async def run(self, context: PhaseContext) -> dict[str, Any]:
        input_slot = _phase_input_slot(self.phase)
        if input_slot not in context.slots:
            input_slot = _legacy_phase_input_slot(self.phase)
        original_input = context.slots.get(input_slot)
        output_slot = _phase_output_slot(self.phase)
        original_output = context.slots.get(output_slot)
        before = dict(context.slots)
        frame = PluginPhaseFrame(
            input=original_input,
            slots=dict(context.slots),
            output=original_output,
        )
        result = self.delegate.run(frame)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Mapping):
            updates = dict(result)
            missing = set(self.declared_produces).difference(updates)
            if missing:
                raise RuntimeError(
                    f"插件 PhaseModule {self.slot} 未产生声明的 slot: "
                    + ", ".join(sorted(missing))
                )
            return updates
        if not isinstance(result, PluginPhaseFrame):
            raise TypeError(
                "插件 PhaseModule 必须返回 Mapping 或 PluginPhaseFrame: "
                f"{type(result).__name__}"
            )
        updates = {
            key: value
            for key, value in result.slots.items()
            if key not in before or not _same_value(before[key], value)
        }
        if not _same_value(original_input, result.input):
            updates[input_slot] = result.input
            if self.phase is LifecyclePhase.BEFORE_TURN:
                ctx = context.slots.get("session:ctx")
                content = getattr(result.input, "content", None)
                if ctx is not None and isinstance(content, str):
                    cast(Any, ctx).content = content
        if not _same_value(original_output, result.output):
            updates[output_slot] = result.output
        return updates


@dataclass(slots=True)
class _LoadedPlugin:
    plugin_id: str
    module_path: str
    instance: Any
    tool_names: tuple[str, ...]


class PluginManager:
    """扫描并按原型顺序直接装配本地 Python 插件。"""

    def __init__(
        self,
        plugin_dirs: list[Path],
        *,
        event_bus: Any = None,
        tool_registry: ToolRegistry | None = None,
        workspace: Path | None = None,
        session_manager: Any = None,
        memory_engine: Any = None,
    ) -> None:
        self._dirs = [path.resolve() for path in plugin_dirs]
        self._event_bus = event_bus
        self._tool_registry = tool_registry or ToolRegistry()
        self._workspace = workspace
        self._plugin_state_root = _plugin_state_root(workspace)
        self._session_manager = session_manager
        self._memory_engine = memory_engine
        self._loaded: list[_LoadedPlugin] = []
        self._tool_hooks: list[ToolHook] = []
        self._phase_modules: list[PhaseModule] = []
        self._prompt_blocks: list[PromptBlock] = []
        self._prompt_render_modules: list[object] = []
        self.diagnostics: list[PluginDiagnostic] = []

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    @property
    def loaded_plugin_ids(self) -> tuple[str, ...]:
        return tuple(item.plugin_id for item in self._loaded)

    @property
    def tool_hooks(self) -> tuple[ToolHook, ...]:
        return tuple(self._tool_hooks)

    @property
    def phase_modules(self) -> tuple[PhaseModule, ...]:
        return tuple(self._phase_modules)

    @property
    def prompt_blocks(self) -> tuple[PromptBlock, ...]:
        return tuple(self._prompt_blocks)

    @property
    def prompt_render_modules(self) -> tuple[object, ...]:
        return tuple(self._prompt_render_modules)

    def get_plugin(self, plugin_id: str) -> Any | None:
        return next(
            (item.instance for item in self._loaded if item.plugin_id == plugin_id),
            None,
        )

    def discover(self) -> list[dict[str, str]]:
        discovered: list[dict[str, str]] = []
        names: set[str] = set()
        for root in self._dirs:
            if not root.is_dir():
                continue
            candidates = (
                [root]
                if (root / "plugin.py").is_file()
                else [path for path in sorted(root.iterdir()) if path.is_dir()]
            )
            for directory in candidates:
                entrypoint = directory / "plugin.py"
                if not directory.is_dir() or not entrypoint.is_file():
                    continue
                if directory.name in names:
                    self.diagnostics.append(
                        PluginDiagnostic(
                            directory.name,
                            "duplicate_plugin",
                            "同名插件按目录优先级跳过",
                        )
                    )
                    continue
                names.add(directory.name)
                digest = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
                discovered.append(
                    {
                        "name": directory.name,
                        "module_path": str(entrypoint.resolve()),
                        "import_path": f"memopilot_plugin_{digest}_{directory.name}",
                    }
                )
        return discovered

    async def load_all(self) -> None:
        for descriptor in self.discover():
            await self._load_one(descriptor)

    async def _load_one(self, descriptor: dict[str, str]) -> None:
        name = descriptor["name"]
        entrypoint = Path(descriptor["module_path"])
        module_path = descriptor["import_path"]
        if (entrypoint.parent / "plugin.disabled").exists():
            return
        if any(item.module_path == module_path for item in self._loaded):
            return
        try:
            self._import_plugin(module_path, entrypoint)
        except Exception as exc:
            self.diagnostics.append(
                PluginDiagnostic(name, "import_failed", f"{type(exc).__name__}: {exc}")
            )
            return
        cls = plugin_registry.get_class(module_path)
        if cls is None:
            self.diagnostics.append(
                PluginDiagnostic(name, "class_missing", "plugin.py 未声明 Plugin 子类")
            )
            return
        instance = cls()
        _apply_manifest(instance, entrypoint.parent)
        plugin_id = str(instance.name or name)
        instance.context = PluginContext(
            event_bus=self._event_bus,
            tool_registry=self._tool_registry,
            plugin_id=plugin_id,
            plugin_dir=entrypoint.parent,
            kv_store=PluginKVStore(
                self._plugin_state_root / f"{_safe_plugin_state_name(plugin_id)}.json"
            ),
            config=_load_plugin_config(entrypoint.parent),
            workspace=self._workspace,
            session_manager=self._session_manager,
            memory_engine=self._memory_engine,
        )
        plugin_registry.register_instance(module_path, instance)
        self._bind_handlers(instance, module_path)
        tool_names = self._register_tools(instance, module_path)
        hook_count_before = len(self._tool_hooks)
        phase_count_before = len(self._phase_modules)
        prompt_block_count_before = len(self._prompt_blocks)
        prompt_render_count_before = len(self._prompt_render_modules)
        self._bind_tool_hooks(instance, module_path)
        self._collect_modules(instance)
        try:
            await instance.initialize()
        except Exception as exc:
            plugin_registry.remove_plugin(module_path)
            for tool_name in tool_names:
                self._tool_registry.unregister(tool_name)
            del self._tool_hooks[hook_count_before:]
            del self._phase_modules[phase_count_before:]
            del self._prompt_blocks[prompt_block_count_before:]
            del self._prompt_render_modules[prompt_render_count_before:]
            self.diagnostics.append(
                PluginDiagnostic(
                    name,
                    "initialization_failed",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return
        self._loaded.append(
            _LoadedPlugin(plugin_id, module_path, instance, tuple(tool_names))
        )

    async def unload_all(self) -> None:
        for loaded in reversed(self._loaded):
            termination_error = await _terminate(loaded.instance)
            if termination_error is not None:
                self.diagnostics.append(
                    PluginDiagnostic(
                        loaded.plugin_id,
                        "termination_failed",
                        f"{type(termination_error).__name__}: {termination_error}",
                    )
                )
            for tool_name in loaded.tool_names:
                self._tool_registry.unregister(tool_name)
            plugin_registry.remove_plugin(loaded.module_path)
        self._loaded.clear()
        self._tool_hooks.clear()
        self._phase_modules.clear()
        self._prompt_blocks.clear()
        self._prompt_render_modules.clear()

    terminate_all = unload_all

    def _bind_handlers(self, instance: Any, module_path: str) -> None:
        for metadata in plugin_registry.get_handlers_by_module_path(module_path):
            if metadata.kind is not MetadataKind.LIFECYCLE:
                continue
            if self._event_bus is None:
                raise RuntimeError("插件生命周期处理器需要 EventBus")
            handler = _build_event_handler(instance, metadata)
            self._event_bus.on(
                handler.event,
                handler.callback,
                observer=handler.observer,
                priority=handler.priority,
                handler_id=handler.handler_id,
            )

    def _register_tools(self, instance: Any, module_path: str) -> list[str]:
        names: list[str] = []
        for metadata in plugin_registry.get_handlers_by_module_path(module_path):
            if metadata.kind is not MetadataKind.TOOL:
                continue
            built = _build_tool(instance, metadata)
            self._tool_registry.register(
                built,
                risk=metadata.tool_risk,
                always_on=metadata.tool_always_on,
                search_hint=metadata.tool_search_hint,
            )
            names.append(built.name)
        return names

    def _bind_tool_hooks(self, instance: Any, module_path: str) -> None:
        for metadata in plugin_registry.get_handlers_by_module_path(module_path):
            if metadata.kind is MetadataKind.TOOL_HOOK:
                self._tool_hooks.append(_build_tool_hook(instance, metadata))

    def _collect_modules(self, instance: Any) -> None:
        for method in _MODULE_METHODS:
            values = _load_module_list(instance, method)
            if method == "prompt_render_modules":
                self._prompt_render_modules.extend(values)
            for index, value in enumerate(values):
                if method == "prompt_render_modules" and isinstance(value, PromptBlock):
                    self._prompt_blocks.append(value)
                    continue
                self._phase_modules.append(
                    _adapt_phase_module(
                        value,
                        _METHOD_PHASES[method],
                        plugin_id=instance.context.plugin_id,
                        provider=method,
                        index=index,
                    )
                )

    @staticmethod
    def _import_plugin(module_path: str, entrypoint: Path) -> None:
        spec = importlib.util.spec_from_file_location(
            module_path,
            entrypoint,
            submodule_search_locations=[str(entrypoint.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载插件文件: {entrypoint}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)

def _build_tool(instance: Any, metadata: PluginHandlerMetadata) -> Tool:
    bound = functools.partial(metadata.handler, instance, None)
    signature = inspect.signature(metadata.handler)
    accepted = frozenset(
        name for name in signature.parameters if name not in {"self", "event"}
    )

    async def execute(**kwargs: Any) -> Any:
        result = bound(**{key: value for key, value in kwargs.items() if key in accepted})
        if inspect.isawaitable(result):
            return await result
        return result

    name = metadata.tool_name or metadata.handler_name
    return Tool(
        name=name,
        description=(metadata.handler.__doc__ or name).strip(),
        parameters=metadata.tool_schema
        or {"type": "object", "properties": {}, "required": []},
        handler=execute,
        source=f"plugin:{getattr(instance, 'name', None) or type(instance).__name__}",
    )


def _build_tool_hook(instance: Any, metadata: PluginHandlerMetadata) -> ToolHook:
    bound = functools.partial(metadata.handler, instance)
    plugin_name = getattr(instance, "name", None) or type(instance).__name__

    class _PluginToolHook(ToolHook):
        def matches(self, context: HookContext) -> bool:
            return (
                metadata.hook_tool_name is None
                or metadata.hook_tool_name == context.request.tool_name
            )

        async def run(self, context: HookContext) -> HookOutcome:
            request = context.request
            result = bound(
                PreToolCtx(
                    tool_name=request.tool_name,
                    arguments=dict(context.current_arguments),
                    session_key=request.session_key,
                    channel=request.channel,
                    chat_id=request.chat_id,
                    call_id=request.call_id,
                    source=request.source,
                    request_text=request.request_text,
                    tool_batch=tuple(request.tool_batch),
                    tool_batch_index=request.tool_batch_index,
                )
            )
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return HookOutcome()
            if isinstance(result, HookOutcome):
                return result
            if isinstance(result, ToolHookDecision):
                return HookOutcome(
                    decision="deny" if result.denied else "pass",
                    updated_input=(
                        dict(result.arguments) if result.arguments is not None else None
                    ),
                    reason=result.reason or "",
                )
            if isinstance(result, dict):
                return HookOutcome(updated_input=cast(dict[str, Any], result))
            raise TypeError(f"插件 ToolHook 返回值无效: {type(result).__name__}")

    hook = _PluginToolHook(
        f"plugin:{plugin_name}:{metadata.handler_name}",
        event="pre_tool_use",
    )

    async def legacy_before(
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolHookDecision | None:
        context = HookContext(
            event="pre_tool_use",
            request=ToolExecutionRequest(
                call_id="",
                tool_name=tool_name,
                arguments=dict(arguments),
            ),
            current_arguments=dict(arguments),
        )
        if not hook.matches(context):
            return None
        outcome = await hook.run(context)
        return ToolHookDecision(
            arguments=outcome.updated_input,
            denied=outcome.decision == "deny",
            reason=outcome.reason or None,
        )

    hook.before = legacy_before
    return hook


def _build_event_handler(instance: Any, metadata: PluginHandlerMetadata) -> EventHandler:
    bound = functools.partial(metadata.handler, instance)

    async def callback(payload: dict[str, object]) -> dict[str, object] | None:
        event_context = PluginEventContext(payload)
        result = bound(event_context)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            return event_context.as_dict() if not metadata.observer else None
        if isinstance(result, PluginEventContext):
            return result.as_dict()
        if not isinstance(result, dict):
            raise TypeError(f"插件事件返回值必须是 dict 或 None: {type(result).__name__}")
        return cast(dict[str, object], result)

    event = metadata.event_type.value if metadata.event_type is not None else ""
    plugin_name = getattr(instance, "name", None) or type(instance).__name__
    return EventHandler(
        handler_id=f"plugin:{plugin_name}:{metadata.handler_name}",
        event=event,
        callback=callback,
        observer=metadata.observer,
        priority=-metadata.priority,
    )


def _load_module_list(instance: Any, method_name: str) -> list[object]:
    provider = getattr(instance, method_name, None)
    if not callable(provider):
        return []
    loaded = provider()
    if loaded is None:
        return []
    if not isinstance(loaded, list):
        raise TypeError(f"{type(instance).__name__}.{method_name} 必须返回 list")
    return loaded


def _load_plugin_config(plugin_dir: Path) -> PluginConfig | None:
    schema_path = plugin_dir / "_conf_schema.json"
    if not schema_path.exists():
        return None
    raw = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("_conf_schema.json 必须是 JSON 对象")
    values = {
        key: spec["default"]
        for key, spec in raw.items()
        if isinstance(key, str) and isinstance(spec, dict) and "default" in spec
    }
    override_path = plugin_dir / "plugin_config.json"
    if override_path.exists():
        override = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise ValueError("plugin_config.json 必须是 JSON 对象")
        values.update({str(key): value for key, value in override.items()})
    return PluginConfig(values)


def _apply_manifest(instance: Any, plugin_dir: Path) -> None:
    path = plugin_dir / "manifest.yaml"
    if not path.exists():
        return
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest.yaml 必须是 YAML 对象")
    for field in ("name", "version", "desc", "author"):
        if raw.get(field) is not None:
            setattr(instance, field, str(raw[field]))


def _is_phase_module(value: object) -> bool:
    return callable(getattr(value, "run", None))


def _adapt_phase_module(
    value: object,
    phase: LifecyclePhase,
    *,
    plugin_id: str,
    provider: str,
    index: int,
) -> PhaseModule:
    if not _is_phase_module(value):
        raise TypeError(f"{phase.value} provider 返回了无效 PhaseModule")
    has_complete_contract = all(
        hasattr(value, attr) for attr in ("phase", "slot", "requires", "produces")
    )
    if has_complete_contract and cast(Any, value).phase is phase:
        return cast(PhaseModule, value)
    slot = str(
        getattr(value, "slot", f"plugin:{plugin_id}:{provider}:{index}")
    )
    requires = tuple(getattr(value, "requires", ()))
    declared_produces = tuple(getattr(value, "produces", ()))
    produces = tuple(item for item in declared_produces if item not in requires)
    return cast(
        PhaseModule,
        _PhaseAdapter(
            phase,
            value,
            slot,
            requires,
            produces,
            declared_produces,
        ),
    )


def _phase_input_slot(phase: LifecyclePhase) -> str:
    return {
        LifecyclePhase.BEFORE_TURN: "before_turn.input",
        LifecyclePhase.BEFORE_REASONING: "before_reasoning.input",
        LifecyclePhase.PROMPT_RENDER: "prompt_render.input",
        LifecyclePhase.BEFORE_STEP: "before_step.input",
        LifecyclePhase.AFTER_STEP: "after_step.input",
        LifecyclePhase.AFTER_REASONING: "after_reasoning.input",
        LifecyclePhase.AFTER_TURN: "after_turn.input",
    }[phase]


def _phase_output_slot(phase: LifecyclePhase) -> str:
    return {
        LifecyclePhase.BEFORE_TURN: "session:ctx",
        LifecyclePhase.BEFORE_REASONING: "reasoning:ctx",
        LifecyclePhase.PROMPT_RENDER: "prompt_render.output",
        LifecyclePhase.BEFORE_STEP: "step:ctx",
        LifecyclePhase.AFTER_STEP: "step:ctx",
        LifecyclePhase.AFTER_REASONING: "after_reasoning.output",
        LifecyclePhase.AFTER_TURN: "turn:ctx",
    }[phase]


def _legacy_phase_input_slot(phase: LifecyclePhase) -> str:
    return {
        LifecyclePhase.BEFORE_TURN: "turn.input",
        LifecyclePhase.BEFORE_REASONING: "reasoning.input",
        LifecyclePhase.PROMPT_RENDER: "reasoning.input",
        LifecyclePhase.BEFORE_STEP: "step.messages",
        LifecyclePhase.AFTER_STEP: "step.response",
        LifecyclePhase.AFTER_REASONING: "reasoning.result",
        LifecyclePhase.AFTER_TURN: "turn.output",
    }[phase]


def _same_value(left: object, right: object) -> bool:
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


async def _terminate(instance: Any) -> Exception | None:
    try:
        result = instance.terminate()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        return exc
    return None


__all__ = ["PluginManager"]
