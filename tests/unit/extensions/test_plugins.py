from __future__ import annotations

import sys
from pathlib import Path

import pytest

from memopilot.extensions.decorators import on_before_turn, on_tool_pre, tool
from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.extensions.prompts import PromptBlock, PromptRenderer
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.engine import TurnInput
from memopilot.runtime.phases import LifecyclePhase, PhaseContext
from memopilot.runtime.tools import Tool, ToolRegistry


def test_prompt_renderer_applies_block_and_total_budgets_deterministically() -> None:
    renderer = PromptRenderer(
        (
            PromptBlock("later", "BBBBBB", priority=20, scopes=("passive",), max_chars=4),
            PromptBlock("first", "AAAA", priority=10, scopes=("all",), max_chars=10),
            PromptBlock("other", "NO", priority=1, scopes=("background",), max_chars=10),
        )
    )

    assert renderer.render(scope="passive", base_prompt="BASE", max_chars=14) == (
        "BASE\n\nAAAA\n\nBB"
    )


def _write_class_plugin(root: Path, plugin_id: str, source: str) -> Path:
    directory = root / plugin_id
    directory.mkdir(parents=True)
    (directory / "plugin.py").write_text(source, encoding="utf-8")
    return directory


@pytest.mark.asyncio
async def test_class_plugin_uses_prototype_lifecycle_context_and_decorators(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_dir = _write_class_plugin(
        tmp_path,
        "weather",
        '''
from memopilot.extensions.decorators import on_before_turn, on_tool_pre, tool
from memopilot.extensions.hooks import ToolHookDecision
from memopilot.extensions.plugin_base import Plugin

class WeatherPlugin(Plugin):
    name = "weather"

    async def initialize(self):
        self.context.kv_store.increment("starts")

    async def terminate(self):
        self.context.kv_store.set("terminated", True)

    @tool("weather_now", risk="read-only", always_on=True, search_hint="天气 预报")
    async def weather(self, event, city: str, days: int = 1):
        """查询天气。\n\n        :param city: 城市\n        :param days: 天数\n        """
        return {"city": city, "days": days}

    @on_before_turn(priority=10)
    async def remember_turn(self, event):
        event["seen"] = True
        return event

    @on_tool_pre(tool_name="weather_now")
    async def normalize_city(self, event):
        arguments = dict(event["arguments"])
        arguments["city"] = arguments["city"].strip()
        return ToolHookDecision(arguments=arguments)
''',
    )
    (plugin_dir / "manifest.yaml").write_text(
        "name: 天气插件\nversion: 2.0\ndesc: 示例\nauthor: MemoPilot\n",
        encoding="utf-8",
    )
    tools = ToolRegistry()
    event_bus = EventBus()
    manager = PluginManager(
        [tmp_path],
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
    )

    await manager.load_all()

    assert manager.loaded_plugin_ids == ("天气插件",)
    instance = manager.get_plugin("天气插件")
    assert instance is not None
    assert instance.name == "天气插件"
    assert instance.version == "2.0"
    assert instance.context.workspace == workspace
    assert instance.context.config is None
    assert instance.context.kv_store.get("starts") == 1
    assert tools.tool_names == ("weather_now",)
    document = tools.get_document("weather_now")
    assert document is not None
    assert document.risk == "read-only"
    assert document.always_on is True
    assert document.search_hint == "天气 预报"
    schema = tools.schemas()[0]["function"]["parameters"]
    assert schema["properties"]["city"]["type"] == "string"
    assert schema["properties"]["days"]["type"] == "number"
    assert schema["required"] == ["city"]
    assert await event_bus.emit("before_turn", {}) == {"seen": True}
    assert len(manager.tool_hooks) == 1

    await manager.unload_all()

    assert instance.context.kv_store.get("terminated") is True
    assert manager.loaded_plugin_ids == ()


@pytest.mark.asyncio
async def test_plugin_kv_is_centralized_under_workspace_and_survives_unload(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_class_plugin(
        tmp_path / "plugins",
        "unsafe",
        '''
from memopilot.extensions.plugin_base import Plugin

class UnsafeNamePlugin(Plugin):
    name = "../../outside"

    async def initialize(self):
        self.context.kv_store.set("kept", True)
''',
    )
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        [tmp_path / "plugins"],
        tool_registry=ToolRegistry(),
        workspace=workspace,
    )

    await manager.load_all()

    state_root = (workspace / ".memopilot" / "plugin_state").resolve()
    state_files = list(state_root.glob("*.json"))
    assert len(state_files) == 1
    assert state_files[0].resolve().parent == state_root
    assert not (plugin_dir / ".kv.json").exists()
    assert manager.get_plugin("../../outside").context.kv_store.get("kept") is True

    await manager.unload_all()

    assert state_files[0].exists()
    assert not (tmp_path / "outside.json").exists()


@pytest.mark.asyncio
async def test_disabled_plugin_and_config_are_compatible_with_prototype(tmp_path: Path) -> None:
    disabled = _write_class_plugin(
        tmp_path,
        "disabled",
        "from memopilot.extensions.plugin_base import Plugin\nclass Disabled(Plugin):\n    pass\n",
    )
    (disabled / "plugin.disabled").write_text("", encoding="utf-8")
    configured = _write_class_plugin(
        tmp_path,
        "configured",
        '''
from memopilot.extensions.plugin_base import Plugin
class Configured(Plugin):
    async def initialize(self):
        assert self.context.config.api_key == "local"
        assert self.context.config.limit == 5
''',
    )
    (configured / "_conf_schema.json").write_text(
        '{"api_key": {"default": "default"}, "limit": {"default": 5}}',
        encoding="utf-8",
    )
    (configured / "plugin_config.json").write_text(
        '{"api_key": "local"}', encoding="utf-8"
    )

    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()

    assert manager.loaded_plugin_ids == ("configured",)


@pytest.mark.asyncio
async def test_initialize_failure_keeps_prototype_event_and_import_semantics(
    tmp_path: Path,
) -> None:
    _write_class_plugin(
        tmp_path,
        "broken",
        '''
from memopilot.extensions.decorators import on_before_turn, on_tool_pre, tool
from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.prompts import PromptBlock
from memopilot.runtime.engine import FunctionPhaseModule
from memopilot.runtime.phases import LifecyclePhase

async def phase(context):
    return {"broken.output": True}

class Broken(Plugin):
    @tool("broken_tool")
    async def broken_tool(self, event):
        return "never"

    @on_before_turn()
    async def before(self, event):
        event["broken_seen"] = True
        return event

    @on_tool_pre(tool_name="broken_tool")
    async def pre(self, event):
        return None

    def before_turn_modules(self):
        return [
            FunctionPhaseModule(
                LifecyclePhase.BEFORE_TURN,
                "broken.phase",
                (),
                ("broken.output",),
                phase,
            )
        ]

    def prompt_render_modules(self):
        return [PromptBlock("broken.prompt", "never")]

    async def initialize(self):
        raise RuntimeError("boom")

    async def terminate(self):
        self.context.kv_store.set("terminated", True)
''',
    )
    tools = ToolRegistry()
    event_bus = EventBus()
    manager = PluginManager([tmp_path], event_bus=event_bus, tool_registry=tools)
    module_path = manager.discover()[0]["import_path"]

    await manager.load_all()

    assert manager.loaded_plugin_ids == ()
    assert tools.tool_names == ()
    assert manager.tool_hooks == ()
    assert manager.phase_modules == ()
    assert manager.prompt_blocks == ()
    assert await event_bus.emit("before_turn", {}) == {"broken_seen": True}
    assert module_path in sys.modules
    state_files = list((tmp_path / ".memopilot" / "plugin_state").glob("*.json"))
    assert state_files == []
    assert manager.diagnostics[0].code == "initialization_failed"


def test_plugin_subclass_and_decorators_register_against_its_module() -> None:
    class InlinePlugin(Plugin):
        @tool("inline")
        async def inline(self, event, value: bool) -> bool:
            return value

        @on_before_turn(priority=3)
        async def before(self, event):
            return event

        @on_tool_pre(tool_name="inline")
        async def pre(self, event):
            return None

    from memopilot.extensions.plugin_registry import plugin_registry

    try:
        assert plugin_registry.get_class(InlinePlugin.__module__) is InlinePlugin
        metadata = plugin_registry.get_handlers_by_module_path(InlinePlugin.__module__)
        assert {item.handler_name for item in metadata} >= {"inline", "before", "pre"}
    finally:
        plugin_registry.remove_plugin(InlinePlugin.__module__)


@pytest.mark.asyncio
async def test_callbacks_receive_prototype_compatible_context_objects(tmp_path: Path) -> None:
    _write_class_plugin(
        tmp_path,
        "contexts",
        '''
from memopilot.extensions.decorators import (
    on_before_turn,
    on_tool_call,
    on_tool_pre,
    on_tool_result,
)
from memopilot.extensions.plugin_base import Plugin

class ContextPlugin(Plugin):
    @on_before_turn()
    async def before(self, event):
        return {"session_key": event.session_key, "seen": True}

    @on_tool_call()
    async def tool_call(self, event):
        self.context.kv_store.set("called", event.tool_name)

    @on_tool_result()
    async def tool_result(self, event):
        self.context.kv_store.set("status", event.status)

    @on_tool_pre(tool_name="shell")
    async def pre(self, event):
        assert event.tool_name == "shell"
        arguments = dict(event.arguments)
        arguments["command"] = "safe"
        return arguments
''',
    )
    event_bus = EventBus()
    manager = PluginManager(
        [tmp_path], event_bus=event_bus, tool_registry=ToolRegistry()
    )
    await manager.load_all()

    assert await event_bus.emit("before_turn", {"session_key": "feishu:1"}) == {
        "session_key": "feishu:1",
        "seen": True,
    }
    await event_bus.observe("before_tool_call", {"tool_name": "shell"})
    await event_bus.observe("after_tool_result", {"status": "success"})
    instance = manager.loaded_plugin_ids[0]
    plugin = manager.get_plugin(instance)
    assert plugin.context.kv_store.get("called") == "shell"
    assert plugin.context.kv_store.get("status") == "success"
    decision = await manager.tool_hooks[0].before("shell", {"command": "rm a"})
    assert decision is not None
    assert decision.arguments == {"command": "safe"}


@pytest.mark.asyncio
async def test_provider_phase_module_is_adapted_to_owning_lifecycle(tmp_path: Path) -> None:
    _write_class_plugin(
        tmp_path,
        "phase",
        '''
from memopilot.extensions.plugin_base import Plugin

class BareModule:
    slot = "plugin.bare"
    requires = ()
    produces = ("plugin.output",)
    async def run(self, frame):
        return {"plugin.output": frame.slots["input"]}

class PhasePlugin(Plugin):
    def before_turn_modules(self):
        return [BareModule()]
''',
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()

    module = manager.phase_modules[0]
    assert module.phase is LifecyclePhase.BEFORE_TURN
    frame = PhaseContext(slots={"input": "ok"})
    assert await module.run(frame) == {"plugin.output": "ok"}


@pytest.mark.asyncio
async def test_initialize_sees_staged_tools_and_failure_unregisters_them(
    tmp_path: Path,
) -> None:
    _write_class_plugin(
        tmp_path,
        "broken_staged",
        '''
from memopilot.extensions.decorators import tool
from memopilot.extensions.plugin_base import Plugin

class BrokenStaged(Plugin):
    @tool("staged")
    async def staged(self, event):
        return "value"
    async def initialize(self):
        assert "staged" in self.context.tool_registry.tool_names
        raise RuntimeError("after staged registration")
''',
    )
    tools = ToolRegistry()
    manager = PluginManager([tmp_path], tool_registry=tools)

    await manager.load_all()

    assert tools.tool_names == ()
    assert manager.diagnostics[0].code == "initialization_failed"
    assert "after staged registration" in manager.diagnostics[0].message


@pytest.mark.asyncio
async def test_unload_keeps_prototype_dynamic_import_semantics(
    tmp_path: Path,
) -> None:
    directory = _write_class_plugin(
        tmp_path,
        "reloadable",
        '''
from .helper import VALUE
from memopilot.extensions.decorators import tool
from memopilot.extensions.plugin_base import Plugin

class Reloadable(Plugin):
    @tool("read_value", risk="read-only")
    async def read_value(self, event):
        return VALUE
''',
    )
    helper = directory / "helper.py"
    helper.write_text('VALUE = "first"\n', encoding="utf-8")
    tools = ToolRegistry()
    manager = PluginManager([tmp_path], tool_registry=tools)
    await manager.load_all()
    first = await tools.execute(FunctionCall("1", "read_value", {}))
    module_path = manager._loaded[0].module_path
    assert first.result == "first"
    assert f"{module_path}.helper" in __import__("sys").modules

    await manager.unload_all()
    assert any(
        name == module_path or name.startswith(f"{module_path}.")
        for name in __import__("sys").modules
    )
    assert tools.tool_names == ()


@pytest.mark.asyncio
async def test_context_includes_injected_session_manager_and_terminate_diagnostic(
    tmp_path: Path,
) -> None:
    _write_class_plugin(
        tmp_path,
        "terminate",
        '''
from memopilot.extensions.plugin_base import Plugin
class TerminatePlugin(Plugin):
    async def initialize(self):
        assert self.context.session_manager == "sessions"
    async def terminate(self):
        raise RuntimeError("cannot close")
''',
    )
    manager = PluginManager(
        [tmp_path], tool_registry=ToolRegistry(), session_manager="sessions"
    )
    await manager.load_all()

    await manager.unload_all()

    assert manager.diagnostics[-1].code == "termination_failed"


@pytest.mark.asyncio
async def test_duplicate_plugin_tool_fails_immediately_and_keeps_core_tool(
    tmp_path: Path,
) -> None:
    async def core_handler() -> str:
        return "core"

    tools = ToolRegistry(
        [Tool("shared", "core", {"type": "object"}, core_handler)]
    )
    _write_class_plugin(
        tmp_path,
        "duplicate",
        '''
from memopilot.extensions.decorators import tool
from memopilot.extensions.plugin_base import Plugin
class DuplicatePlugin(Plugin):
    @tool("shared")
    async def shared(self, event):
        return "plugin"
''',
    )
    manager = PluginManager([tmp_path], tool_registry=tools)

    with pytest.raises(ValueError, match="工具名称重复: shared"):
        await manager.load_all()

    assert tools.tool_names == ("shared",)
    observation = await tools.execute(FunctionCall("core", "shared", {}))
    assert observation.result == "core"


@pytest.mark.asyncio
async def test_real_prototype_module_shape_gets_stable_defaults_and_frame_adapter(
    tmp_path: Path,
) -> None:
    _write_class_plugin(
        tmp_path,
        "prototype_phase",
        '''
from memopilot.extensions.plugin_base import Plugin

class EarlyModule:
    requires = ("turn.input",)
    async def run(self, frame):
        frame.slots["plugin.changed"] = "yes"
        return frame

class LateModule:
    async def run(self, frame):
        return frame

class PromptTopModule:
    async def run(self, frame):
        return frame

class PrototypePhasePlugin(Plugin):
    name = "prototype_phase"
    def before_turn_modules(self):
        return [EarlyModule(), LateModule()]
    def prompt_render_modules(self):
        return [PromptTopModule()]
''',
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())

    await manager.load_all()

    assert [module.phase for module in manager.phase_modules] == [
        LifecyclePhase.BEFORE_TURN,
        LifecyclePhase.BEFORE_TURN,
        LifecyclePhase.PROMPT_RENDER,
    ]
    assert [module.slot for module in manager.phase_modules] == [
        "plugin:prototype_phase:before_turn_modules:0",
        "plugin:prototype_phase:before_turn_modules:1",
        "plugin:prototype_phase:prompt_render_modules:0",
    ]
    assert manager.phase_modules[0].requires == ("turn.input",)
    assert manager.phase_modules[0].produces == ()
    frame = PhaseContext(slots={"turn.input": "hello"})
    assert await manager.phase_modules[0].run(frame) == {"plugin.changed": "yes"}
    assert await manager.phase_modules[1].run(frame) == {}


@pytest.mark.asyncio
async def test_prototype_phase_adapter_uses_real_frame_and_writes_back_mutations(
    tmp_path: Path,
) -> None:
    _write_class_plugin(
        tmp_path,
        "frame_contract",
        '''
from dataclasses import replace
from memopilot.extensions.plugin_base import Plugin

class RewriteInput:
    slot = "frame_contract.rewrite"
    requires = ("turn.input",)
    produces = ("plugin.changed",)

    async def run(self, frame):
        assert type(frame).__name__ == "PluginPhaseFrame"
        frame.input = replace(frame.input, content="插件改写后的问题")
        frame.slots["plugin.changed"] = frame.input.content
        frame.output = "插件阶段输出"
        return frame

class FrameContractPlugin(Plugin):
    def before_turn_modules(self):
        return [RewriteInput()]
''',
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()
    module = manager.phase_modules[0]
    turn = TurnInput(session_key="feishu:1", content="原始问题")
    context = PhaseContext(slots={"turn.input": turn})

    updates = await module.run(context)

    assert updates["turn.input"].content == "插件改写后的问题"
    assert updates["plugin.changed"] == "插件改写后的问题"
    assert updates["session:ctx"] == "插件阶段输出"


@pytest.mark.asyncio
async def test_prototype_phase_adapter_rejects_invalid_run_result(tmp_path: Path) -> None:
    _write_class_plugin(
        tmp_path,
        "invalid_frame",
        '''
from memopilot.extensions.plugin_base import Plugin

class InvalidFrame:
    async def run(self, frame):
        return "not-a-frame"

class InvalidFramePlugin(Plugin):
    def before_turn_modules(self):
        return [InvalidFrame()]
''',
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()

    with pytest.raises(TypeError, match="Mapping 或 PluginPhaseFrame"):
        await manager.phase_modules[0].run(
            PhaseContext(slots={"turn.input": TurnInput("feishu:1", "问题")})
        )


@pytest.mark.asyncio
async def test_prototype_phase_adapter_allows_conditional_frame_exports(tmp_path: Path) -> None:
    _write_class_plugin(
        tmp_path,
        "conditional_frame",
        '''
from memopilot.extensions.plugin_base import Plugin

class ConditionalExport:
    produces = ("step:early_stop_reason",)

    async def run(self, frame):
        return frame

class ConditionalFramePlugin(Plugin):
    def after_step_modules(self):
        return [ConditionalExport()]
''',
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()

    assert await manager.phase_modules[0].run(PhaseContext()) == {}


@pytest.mark.asyncio
async def test_prototype_phase_adapter_keeps_mapping_exports_strict(tmp_path: Path) -> None:
    _write_class_plugin(
        tmp_path,
        "strict_mapping",
        '''
from memopilot.extensions.plugin_base import Plugin

class MissingMappingExport:
    produces = ("plugin:required",)

    async def run(self, frame):
        return {}

class StrictMappingPlugin(Plugin):
    def before_turn_modules(self):
        return [MissingMappingExport()]
''',
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()

    with pytest.raises(RuntimeError, match="plugin:required"):
        await manager.phase_modules[0].run(PhaseContext())
