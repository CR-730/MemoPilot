from __future__ import annotations

import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from memopilot.bus.events import InboundMessage, TurnCommitted
from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
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
from memopilot.runtime.tools import Tool, ToolRegistry

_BUILTIN_ROOT = Path(__file__).parents[3] / "src" / "memopilot" / "builtin_plugins"
_STATUS_DIR = _BUILTIN_ROOT / "status_commands"
_OBSERVE_DIR = _BUILTIN_ROOT / "observe"
_NOW = datetime(2026, 7, 29, 1, 2, tzinfo=UTC)


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
    def __init__(self, responses: Sequence[ModelResponse] = ()) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        del messages, tools
        self.calls += 1
        return self.responses.pop(0)


async def _load_status(
    tmp_path: Path,
    repository: ConversationRepository,
) -> PluginManager:
    manager = PluginManager(
        [_STATUS_DIR],
        tool_registry=ToolRegistry(),
        workspace=tmp_path,
        session_manager=repository,
    )
    await manager.load_all()
    assert manager.loaded_plugin_ids == ("status_commands",)
    return manager


def _commit_turn(
    repository: ConversationRepository,
    index: int,
    user: str,
    assistant: str,
) -> None:
    repository.commit_turn(
        InboundMessage(
            "cli",
            "user",
            "chat",
            user,
            timestamp=_NOW,
            metadata={"message_id": f"message-{index}"},
        ),
        assistant_content=assistant,
    )


async def test_memory_status_short_circuits_provider_and_reads_repository(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = ConversationRepository(database)
    _commit_turn(repository, 1, "第一个问题", "第一个回答")
    _commit_turn(repository, 2, "第二个问题", "第二个回答")
    with connect_database(database) as connection:
        connection.execute(
            "UPDATE sessions SET last_consolidated_position = 2 WHERE session_key = 'cli:chat'"
        )
        connection.commit()
    manager = await _load_status(tmp_path, repository)
    provider = _Provider()

    runtime = DefaultReasoner(
        provider,
        ToolRegistry(),
        modules=manager.phase_modules,
    )
    runtime._test_outer_modules = manager.phase_modules  # type: ignore[attr-defined]
    result = await run_through_passive_pipeline(
        runtime,
        TurnInput("cli:chat", "/memorystatus"),
    )

    assert provider.calls == 0
    assert result.react.exit_reason == "before_turn_abort"
    assert "最后已整理的用户消息：" in result.reply
    assert "“第一个问题”" in result.reply
    assert "尚未整理的用户消息数：1" in result.reply
    assert "当前会话消息数：4" in result.reply
    await manager.unload_all()


async def test_real_cache_usage_is_aggregated_observed_and_displayed(
    tmp_path: Path,
) -> None:
    async def echo(*, text: str) -> str:
        return text

    provider = _Provider(
        (
            ModelResponse(
                content=None,
                tool_calls=(FunctionCall("call-1", "echo", {"text": "ok"}),),
                prompt_tokens=100,
                prompt_cache_hit_tokens=80,
            ),
            ModelResponse(
                content="最终回答",
                tool_calls=(),
                prompt_tokens=50,
                prompt_cache_hit_tokens=40,
            ),
        )
    )
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
    turn = await run_through_passive_pipeline(
        DefaultReasoner(provider, tools), TurnInput("cli:chat", "执行")
    )
    assert turn.react.prompt_tokens == 150
    assert turn.react.prompt_cache_hit_tokens == 120

    bus = EventBus()
    observe = PluginManager(
        [_OBSERVE_DIR],
        event_bus=bus,
        tool_registry=ToolRegistry(),
        workspace=tmp_path,
    )
    await observe.load_all()
    await bus.fanout(
        TurnCommitted(
            "cli:chat",
            "cli",
            "chat",
            "执行",
            turn.reply,
            ["echo"],
            _NOW,
            react_cache_prompt_tokens=turn.react.prompt_tokens,
            react_cache_hit_tokens=turn.react.prompt_cache_hit_tokens,
        )
    )
    await observe.unload_all()

    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = ConversationRepository(database)
    status = await _load_status(tmp_path, repository)
    blocked_provider = _Provider()
    runtime = DefaultReasoner(
        blocked_provider,
        ToolRegistry(),
        modules=status.phase_modules,
    )
    runtime._test_outer_modules = status.phase_modules  # type: ignore[attr-defined]
    result = await run_through_passive_pipeline(
        runtime,
        TurnInput("cli:chat", "/kvcache 5"),
    )

    assert blocked_provider.calls == 0
    assert "命中率  80.0%" in result.reply
    assert "Token  120 / 150" in result.reply
    assert "最终回答" in result.reply
    await status.unload_all()
    await bus.aclose()
