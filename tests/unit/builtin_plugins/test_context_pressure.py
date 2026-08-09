from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_events import AfterStepCtx
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
from memopilot.runtime.phases import LifecyclePhase, PluginPhaseFrame
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import Tool, ToolRegistry

PLUGIN_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "memopilot"
    / "builtin_plugins"
    / "context_pressure"
)


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
        self.calls = 0

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        del messages, tools
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="working",
                tool_calls=(FunctionCall("c1", "echo", {"text": "done"}),),
                finish_reason="tool_calls",
            )
        return ModelResponse(content="must not run", tool_calls=())


async def test_context_pressure_stops_only_above_eighty_percent_with_more_work() -> None:
    manager = PluginManager([PLUGIN_DIR], tool_registry=ToolRegistry())
    await manager.load_all()
    module = manager.get_plugin("context_pressure").after_step_modules()[0]

    async def apply_pressure(tokens: int, *, has_more: bool) -> AfterStepCtx:
        ctx = AfterStepCtx(
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            iteration=1,
            context_tokens_estimate=tokens,
            tools_called=(),
            partial_reply="",
            tools_used_so_far=(),
            tool_chain_partial=(),
            partial_thinking=None,
            has_more=has_more,
            context_window_tokens=100,
        )
        await module.run(PluginPhaseFrame(input=ctx, slots={"step:ctx": ctx}))
        return ctx

    assert not (await apply_pressure(80, has_more=True)).early_stop
    assert not (await apply_pressure(81, has_more=False)).early_stop
    over_limit = await apply_pressure(81, has_more=True)
    assert over_limit.early_stop
    assert over_limit.early_stop_reason == "context_pressure"

    calls: list[str] = []

    async def echo(text: str) -> str:
        calls.append(text)
        return text

    tools = ToolRegistry(
        [
            Tool(
                "echo",
                "echo",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                echo,
            )
        ]
    )
    provider = _Provider()
    result = await run_through_passive_pipeline(
        DefaultReasoner(
            provider,
            tools,
            modules=manager.phase_modules,
            context_window_tokens=1,
        ),
        TurnInput("feishu:chat-1", "question"),
    )

    assert calls == ["done"]
    assert provider.calls == 1
    assert result.react.exit_reason == "context_pressure"
