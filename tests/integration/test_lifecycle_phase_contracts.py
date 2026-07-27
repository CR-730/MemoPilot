from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import ToolRegistry


class _CapturingProvider(ChatProvider):
    def __init__(
        self,
        reply: str = "provider reply",
        *,
        thinking: str | None = None,
    ) -> None:
        self.reply = reply
        self.thinking = thinking
        self.calls: list[tuple[ChatMessage, ...]] = []

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        del tools
        self.calls.append(tuple(messages))
        return ModelResponse(
            content=self.reply,
            tool_calls=(),
            finish_reason="stop",
            thinking=self.thinking,
            provider_fields=(
                {"reasoning_content": self.thinking}
                if self.thinking is not None
                else {}
            ),
        )


def _write_all_phase_plugin(root: Path) -> None:
    plugin_dir = root / "all_phases"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(
        '''
from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import (
    AfterReasoningInput,
    AfterReasoningResult,
    AfterStepCtx,
    AfterTurnInput,
    BeforeReasoningInput,
    BeforeStepInput,
    BeforeTurnInput,
    PromptRenderInput,
)
from memopilot.extensions.prompts import PromptRenderContext, PromptSectionRender

class BeforeTurnModule:
    slot = "all.before_turn"
    requires = ("before_turn.build_ctx", "session:ctx")
    produces = ("session:ctx",)
    async def run(self, frame):
        assert isinstance(frame.input, BeforeTurnInput)
        ctx = frame.slots["session:ctx"]
        ctx.content = "rewritten question"
        return frame

class BeforeReasoningModule:
    slot = "all.before_reasoning"
    requires = ("before_reasoning.build_ctx", "reasoning:ctx")
    produces = ("reasoning:ctx",)
    async def run(self, frame):
        assert isinstance(frame.input, BeforeReasoningInput)
        frame.slots["reasoning:ctx"].extra_hints.append("outer hint")
        return frame

class PromptModule:
    slot = "all.prompt"
    requires = ("prompt_render.emit", "prompt:ctx")
    produces = ("prompt:ctx",)
    async def run(self, frame):
        assert isinstance(frame.input, PromptRenderInput)
        assert isinstance(frame.slots["prompt:ctx"], PromptRenderContext)
        frame.slots["prompt:ctx"].system_sections_bottom.append(
            PromptSectionRender("plugin", "prompt plugin", is_static=False)
        )
        return frame

class BeforeStepModule:
    slot = "all.before_step"
    requires = ("before_step.emit", "step:ctx")
    produces = ("step:ctx",)
    async def run(self, frame):
        assert isinstance(frame.input, BeforeStepInput)
        frame.slots["step:ctx"].extra_hints.append("step hint")
        return frame

class AfterStepModule:
    slot = "all.after_step"
    requires = ("after_step.copy_input", "step:ctx")
    produces = ("step:ctx",)
    async def run(self, frame):
        assert isinstance(frame.input, AfterStepCtx)
        frame.input.extra_metadata["observed"] = True
        return frame

class AfterReasoningModule:
    slot = "all.after_reasoning"
    requires = ("after_reasoning.build_ctx", "reasoning:ctx")
    produces = ("reasoning:ctx",)
    async def run(self, frame):
        assert isinstance(frame.input, AfterReasoningInput)
        ctx = frame.slots["reasoning:ctx"]
        ctx.reply = "plugin reply"
        ctx.media.append("image.png")
        ctx.outbound_metadata["plugin"] = "kept"
        return frame

class AfterTurnModule:
    slot = "all.after_turn"
    requires = ("after_turn.build_ctx", "turn:ctx")
    produces = ("turn:ctx",)
    def __init__(self, plugin):
        self.plugin = plugin
    async def run(self, frame):
        assert isinstance(frame.input, AfterTurnInput)
        self.plugin.context.kv_store.set("after_turn", frame.slots["turn:ctx"].reply)
        return frame

class AllPhasesPlugin(Plugin):
    name = "all_phases"
    def before_turn_modules(self): return [BeforeTurnModule()]
    def before_reasoning_modules(self): return [BeforeReasoningModule()]
    def prompt_render_modules(self): return [PromptModule()]
    def before_step_modules(self): return [BeforeStepModule()]
    def after_step_modules(self): return [AfterStepModule()]
    def after_reasoning_modules(self): return [AfterReasoningModule()]
    def after_turn_modules(self): return [AfterTurnModule(self)]
''',
        encoding="utf-8",
    )


async def test_all_seven_prototype_phase_frames_flow_through_real_runtime(
    tmp_path: Path,
) -> None:
    _write_all_phase_plugin(tmp_path)
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()
    provider = _CapturingProvider()

    result = await AgentRuntime(
        provider,
        ToolRegistry(),
        modules=manager.phase_modules,
    ).run(
        TurnInput(
            "feishu:chat-1",
            "original question",
            system_prompt="core",
            media=("input.png",),
            outbound_metadata={"request": "kept"},
        )
    )

    rewritten = [item.content for item in provider.calls[0] if item.role == "user"][-1]
    assert (rewritten or "").endswith("rewritten question")
    assert "outer hint" in (provider.calls[0][0].content or "")
    assert "prompt plugin" in (provider.calls[0][0].content or "")
    assert any("step hint" in (item.content or "") for item in provider.calls[0])
    assert result.reply == "plugin reply"
    assert result.messages[-1].content == "plugin reply"
    assert result.media == ("image.png",)
    assert result.outbound_metadata == {"request": "kept", "plugin": "kept"}
    plugin = manager.get_plugin("all_phases")
    assert plugin.context.kv_store.get("after_turn") == "plugin reply"


async def test_before_step_gate_stops_before_provider_and_preserves_phase_fields() -> None:
    provider = _CapturingProvider()
    bus = EventBus()
    seen: dict[str, object] = {}

    async def before_turn(payload):
        payload["content"] = "gate question"
        return payload

    async def before_reasoning(payload):
        payload["extra_hints"].append("reasoning gate hint")
        return payload

    async def before_step(payload):
        seen.update(payload)
        payload["extra_hints"].append("step gate hint")
        payload["early_stop"] = True
        payload["early_stop_reply"] = "stopped safely"
        return payload

    bus.on("before_turn", before_turn)
    bus.on("before_reasoning", before_reasoning)
    bus.on("before_step", before_step)

    result = await AgentRuntime(provider, ToolRegistry(), event_bus=bus).run(
        TurnInput("feishu:chat-1", "original")
    )

    assert provider.calls == []
    assert seen["session_key"] == "feishu:chat-1"
    assert seen["channel"] == "feishu"
    assert seen["chat_id"] == "chat-1"
    assert seen["visible_tool_names"] == frozenset()
    assert result.reply == "stopped safely"
    assert result.messages[-1].content == "stopped safely"
    assert result.react.exit_reason == "early_stop"


async def test_after_reasoning_plugin_receives_generic_final_thinking_without_reply_leak(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "thinking_observer"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(
        '''
from memopilot.extensions.plugin_base import Plugin

class CaptureThinking:
    slot = "thinking_observer.capture"
    requires = ("after_reasoning.build_ctx", "reasoning:ctx")
    produces = ("reasoning:ctx",)
    def __init__(self, plugin):
        self.plugin = plugin
    async def run(self, frame):
        self.plugin.context.kv_store.set("thinking", frame.slots["reasoning:ctx"].thinking)
        return frame

class ThinkingObserverPlugin(Plugin):
    name = "thinking_observer"
    def after_reasoning_modules(self):
        return [CaptureThinking(self)]
''',
        encoding="utf-8",
    )
    manager = PluginManager([tmp_path], tool_registry=ToolRegistry())
    await manager.load_all()
    provider = _CapturingProvider("visible answer", thinking="private chain")

    result = await AgentRuntime(
        provider,
        ToolRegistry(),
        modules=manager.phase_modules,
    ).run(TurnInput("feishu:chat-1", "question"))

    plugin = manager.get_plugin("thinking_observer")
    assert plugin.context.kv_store.get("thinking") == "private chain"
    assert result.react.thinking == "private chain"
    assert result.reply == "visible answer"
    assert result.messages[-1].content == "visible answer"
    assert "private chain" not in result.reply
