from __future__ import annotations

from pathlib import Path

from memopilot.extensions.plugins import PluginRuntime
from memopilot.extensions.prompts import PromptBlock, PromptRenderer


def _write_plugin(
    root: Path,
    plugin_id: str,
    source: str,
    *,
    requires: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = ("tools",),
) -> None:
    directory = root / plugin_id
    directory.mkdir(parents=True)
    dependencies = "\n".join(f"  - {item}" for item in requires) or "  []"
    declared = "\n".join(f"  - {item}" for item in capabilities) or "  []"
    (directory / "manifest.yaml").write_text(
        f"""
plugin_id: {plugin_id}
version: 1.0.0
api_version: 1
entrypoint: plugin.py:register
requires:
{dependencies}
capabilities:
{declared}
""",
        encoding="utf-8",
    )
    (directory / "plugin.py").write_text(source, encoding="utf-8")


def test_plugin_contributions_commit_atomically_and_render_prompt(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "demo",
        """
from memopilot.extensions.prompts import PromptBlock
from memopilot.runtime.tools import Tool

async def ping():
    return "pong"

def register(context):
    context.register_tool(Tool("demo_ping", "ping", {"type": "object"}, ping))
    context.register_prompt_block(
        PromptBlock("demo.identity", "插件身份", priority=20, scopes=("passive",), max_chars=20)
    )
""",
        capabilities=("tools", "prompt_blocks"),
    )

    loaded = PluginRuntime().load_directory(tmp_path)

    assert loaded.enabled_plugin_ids == ("demo",)
    assert [tool.name for tool in loaded.registry.tools] == ["demo_ping"]
    rendered = PromptRenderer(loaded.registry.prompt_blocks).render(
        scope="passive",
        base_prompt="核心提示",
        max_chars=20,
    )
    assert rendered == "核心提示\n\n插件身份"


def test_failed_plugin_leaves_no_partial_registration(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "broken",
        """
from memopilot.runtime.tools import Tool

async def ping():
    return "pong"

def register(context):
    context.register_tool(Tool("broken_ping", "ping", {"type": "object"}, ping))
    raise RuntimeError("boom")
""",
    )

    loaded = PluginRuntime().load_directory(tmp_path)

    assert loaded.enabled_plugin_ids == ()
    assert loaded.registry.tools == ()
    assert loaded.diagnostics[0].plugin_id == "broken"
    assert loaded.diagnostics[0].code == "registration_failed"
    assert "boom" in loaded.diagnostics[0].message


def test_missing_dependency_disables_only_dependent_plugin(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "base", "def register(context):\n    pass\n", capabilities=())
    _write_plugin(
        tmp_path,
        "dependent",
        "def register(context):\n    pass\n",
        requires=("missing",),
        capabilities=(),
    )

    loaded = PluginRuntime().load_directory(tmp_path)

    assert loaded.enabled_plugin_ids == ("base",)
    assert [(item.plugin_id, item.code) for item in loaded.diagnostics] == [
        ("dependent", "missing_dependency")
    ]


def test_duplicate_tool_disables_later_plugin_without_polluting_registry(
    tmp_path: Path,
) -> None:
    source = """
from memopilot.runtime.tools import Tool

async def ping():
    return "pong"

def register(context):
    context.register_tool(Tool("shared", "ping", {"type": "object"}, ping))
"""
    _write_plugin(tmp_path, "a", source)
    _write_plugin(tmp_path, "b", source)

    loaded = PluginRuntime().load_directory(tmp_path)

    assert loaded.enabled_plugin_ids == ("a",)
    assert [tool.name for tool in loaded.registry.tools] == ["shared"]
    assert [(item.plugin_id, item.code) for item in loaded.diagnostics] == [
        ("b", "registration_failed")
    ]


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


def test_invalid_phase_dependency_disables_plugin_before_runtime_start(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "bad_phase",
        """
from memopilot.runtime.engine import FunctionPhaseModule
from memopilot.runtime.phases import LifecyclePhase

async def run(context):
    return {"plugin.output": True}

def register(context):
    context.register_phase_module(
        FunctionPhaseModule(
            LifecyclePhase.AFTER_TURN,
            "after_turn.bad",
            ("missing.slot",),
            ("plugin.output",),
            run,
        )
    )
""",
        capabilities=("phase_modules",),
    )

    loaded = PluginRuntime().load_directory(tmp_path)

    assert loaded.enabled_plugin_ids == ()
    assert loaded.registry.phase_modules == ()
    assert loaded.diagnostics[0].code == "registration_failed"
    assert "missing_dependency" in loaded.diagnostics[0].message
