from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    ToolSchema,
)
from memopilot.runtime.engine import DefaultReasoner, TurnInput
from memopilot.runtime.passive_turn import PassiveTurnPipeline
from memopilot.runtime.phases import LifecyclePhase
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import Tool, ToolExecutor, ToolRegistry


class _NoopOutbound:
    async def dispatch(self, dispatch):
        del dispatch
        return True


async def run_through_passive_pipeline(
    reasoner: DefaultReasoner, turn: TurnInput, **kwargs: object
):
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "operational.db"
        migrate_database(database, DatabaseKind.OPERATIONAL)
        pipeline = PassiveTurnPipeline(
            reasoner,
            repository=ConversationRepository(database),
            outbound=_NoopOutbound(),
            event_bus=reasoner._event_bus or EventBus(),
            history_limit=50,
            after_turn_modules=tuple(
                module
                for module in getattr(reasoner, "_test_outer_modules", ())
                if module.phase is LifecyclePhase.AFTER_TURN
            ),
            outer_modules=tuple(
                module
                for module in getattr(reasoner, "_test_outer_modules", ())
                if module.phase
                in {
                    LifecyclePhase.BEFORE_TURN,
                    LifecyclePhase.BEFORE_REASONING,
                    LifecyclePhase.AFTER_REASONING,
                }
            ),
        )
        return await pipeline.execute_direct(turn, **kwargs)


class _Provider(ChatProvider):
    def __init__(self) -> None:
        self._responses = [
            ModelResponse(
                content=None,
                tool_calls=(FunctionCall("c1", "shell", {"command": "rm a"}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="完成", tool_calls=(), finish_reason="stop"),
        ]

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        del messages, tools
        return self._responses.pop(0)


class _CapturingProvider(ChatProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[ChatMessage, ...]] = []

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        del tools
        self.calls.append(tuple(messages))
        return ModelResponse(content="完成", tool_calls=(), finish_reason="stop")


def _write_plugin(root: Path) -> None:
    plugin_dir = root / "runtime_audit"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(
        """
from memopilot.extensions.decorators import (
    on_after_step,
    on_before_turn,
    on_tool_call,
    on_tool_pre,
    on_tool_result,
)
from memopilot.extensions.plugin_base import Plugin

class RuntimeAudit(Plugin):
    @on_before_turn()
    async def before_turn(self, event):
        self.context.kv_store.set("turn", [event.session_key, event.content])

    @on_after_step()
    async def after_step(self, event):
        self.context.kv_store.set("iteration", event.iteration)

    @on_tool_call()
    async def tool_call(self, event):
        self.context.kv_store.set("call", [event.tool_name, event.status])

    @on_tool_result()
    async def tool_result(self, event):
        self.context.kv_store.set("result", [event.tool_name, event.status])

    @on_tool_pre(tool_name="shell")
    async def tool_pre(self, event):
        assert event.session_key == "feishu:chat-1"
        assert event.source == "passive"
        return {"command": "safe"}
""",
        encoding="utf-8",
    )


async def test_plugin_decorators_fire_through_real_agent_runtime(tmp_path: Path) -> None:
    _write_plugin(tmp_path)
    received: list[str] = []

    async def shell(command: str) -> str:
        received.append(command)
        return "ok"

    tools = ToolRegistry(
        [
            Tool(
                "shell",
                "shell",
                {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                shell,
            )
        ]
    )
    bus = EventBus()
    manager = PluginManager([tmp_path], event_bus=bus, tool_registry=tools)
    await manager.load_all()
    executor = ToolExecutor(tools, manager.tool_hooks)

    result = await run_through_passive_pipeline(
        DefaultReasoner(_Provider(), tools, event_bus=bus, tool_executor=executor),
        TurnInput(session_key="feishu:chat-1", content="删除 a"),
    )

    plugin = manager.get_plugin("runtime_audit")
    assert result.reply == "完成"
    assert received == ["safe"]
    assert plugin.context.kv_store.get("turn") == ["feishu:chat-1", "删除 a"]
    assert plugin.context.kv_store.get("iteration") == 2
    assert plugin.context.kv_store.get("call") == ["shell", "started"]
    assert plugin.context.kv_store.get("result") == ["shell", "success"]

    await manager.unload_all()
    await bus.aclose()


async def test_prototype_phase_and_prompt_modules_change_real_runtime_input(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "prototype_runtime"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(
        """
from dataclasses import replace
from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.prompts import PromptRenderContext, PromptSectionRender

class RewriteBeforeTurn:
    slot = "prototype_runtime.rewrite"
    requires = ("turn.input",)
    produces = ("plugin.rewritten",)

    async def run(self, frame):
        frame.input = replace(frame.input, content=f"已改写：{frame.input.content}")
        frame.slots["plugin.rewritten"] = True
        return frame

class DynamicPrompt:
    slot = "prototype_runtime.prompt"
    requires = ("prompt_render.emit", "prompt:ctx")
    produces = ("prompt:ctx",)

    async def run(self, frame):
        ctx = frame.slots["prompt:ctx"]
        assert isinstance(ctx, PromptRenderContext)
        ctx.system_sections_top.append(
            PromptSectionRender("identity", "插件身份", is_static=True)
        )
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                "turn",
                f"当前会话={ctx.session_key};当前问题={frame.input.content}",
                is_static=False,
            )
        )
        return frame

class PrototypeRuntimePlugin(Plugin):
    name = "prototype_runtime"

    def before_turn_modules(self):
        return [RewriteBeforeTurn()]

    def prompt_render_modules(self):
        return [DynamicPrompt()]
""",
        encoding="utf-8",
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()
    provider = _CapturingProvider()
    runtime = DefaultReasoner(
        provider,
        ToolRegistry(),
        modules=manager.phase_modules,
    )
    runtime._test_outer_modules = manager.phase_modules  # type: ignore[attr-defined]

    await run_through_passive_pipeline(
        runtime, TurnInput("feishu:user-a", "甲的问题", system_prompt="核心")
    )
    await run_through_passive_pipeline(
        runtime, TurnInput("feishu:user-b", "乙的问题", system_prompt="核心")
    )

    first, second = provider.calls
    assert (first[-1].content or "").endswith("已改写：甲的问题")
    assert (second[-1].content or "").endswith("已改写：乙的问题")
    assert first[0].content == (
        "插件身份\n\n核心\n\n当前会话=feishu:user-a;当前问题=已改写：甲的问题"
    )
    assert second[0].content == (
        "插件身份\n\n核心\n\n当前会话=feishu:user-b;当前问题=已改写：乙的问题"
    )
    assert "user-a" not in (second[0].content or "")
