from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.contracts import ChatMessage, ModelResponse
from memopilot.runtime.engine import AgentRuntime, SessionHistoryRequest, TurnInput
from memopilot.runtime.session import OperationalSessionManager
from memopilot.runtime.tools import ToolRegistry


def test_session_history_request_is_typed() -> None:
    assert SessionHistoryRequest("feishu:chat-1", 20).limit == 20


class _SessionManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def get_history(self, session_key: str, limit: int) -> tuple[ChatMessage, ...]:
        self.calls.append((session_key, limit))
        return (ChatMessage.user("earlier"),)


class _Provider:
    def __init__(self) -> None:
        self.messages: tuple[ChatMessage, ...] = ()

    async def complete(self, *, messages: Sequence[ChatMessage], tools: object) -> ModelResponse:
        del tools
        self.messages = tuple(messages)
        return ModelResponse(content="ok", tool_calls=(), finish_reason="stop")


@pytest.mark.asyncio
async def test_before_turn_resolves_session_history_once() -> None:
    manager = _SessionManager()
    provider = _Provider()
    runtime = AgentRuntime(provider, ToolRegistry(), session_manager=manager)

    await runtime.run(
        TurnInput("feishu:chat-1", "current", history=SessionHistoryRequest("feishu:chat-1", 20))
    )

    assert manager.calls == [("feishu:chat-1", 20)]
    messages = [message for message in provider.messages if message.role != "system"]
    assert [(message.role, message.content) for message in messages[:-1]] == [("user", "earlier")]
    assert messages[-1].role == "user"
    assert (messages[-1].content or "").endswith("current")


@pytest.mark.asyncio
async def test_explicit_history_does_not_query_session_manager() -> None:
    manager = _SessionManager()
    runtime = AgentRuntime(_Provider(), ToolRegistry(), session_manager=manager)

    await runtime.run(TurnInput("feishu:chat-1", "current", history=()))

    assert manager.calls == []


@pytest.mark.asyncio
async def test_session_history_request_requires_manager() -> None:
    runtime = AgentRuntime(_Provider(), ToolRegistry())

    with pytest.raises(RuntimeError, match="SessionHistoryRequest requires SessionManager"):
        await runtime.run(
            TurnInput(
                "feishu:chat-1",
                "current",
                history=SessionHistoryRequest("feishu:chat-1", 20),
            )
        )


@pytest.mark.asyncio
async def test_before_turn_expands_persisted_tool_history_for_provider(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = ConversationRepository(database)
    now = datetime(2026, 7, 30, tzinfo=UTC)
    from memopilot.bus.events import InboundMessage

    message = InboundMessage("cli", "user", "chat", "first", timestamp=now)
    repository.record_inbound_activity(message)
    repository.commit_turn(
        message,
        assistant_content="final",
        assistant_tool_chain=(
            {
                "text": "calling",
                "calls": [
                    {"call_id": "ok", "name": "tool", "arguments": {}, "result": "ok-result"},
                    {"call_id": "bad", "name": "tool", "arguments": {}, "error": "bad-result"},
                ],
            },
        ),
    )

    class CountingManager(OperationalSessionManager):
        def __init__(self, repository) -> None:  # type: ignore[no-untyped-def]
            super().__init__(repository)
            self.calls = []

        def get_history(self, session_key: str, limit: int) -> tuple[ChatMessage, ...]:
            self.calls.append((session_key, limit))
            return super().get_history(session_key, limit)

    manager = CountingManager(repository)
    provider = _Provider()
    runtime = AgentRuntime(provider, ToolRegistry(), session_manager=manager)

    await runtime.run(
        TurnInput("cli:chat", "current", history=SessionHistoryRequest("cli:chat", 20))
    )

    history = [item for item in provider.messages if item.role != "system"]
    assert [item.role for item in history] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "user",
    ]
    assert [item.tool_call_id for item in history[2:4]] == ["ok", "bad"]
    assert history[4].content == "final"
    assert (history[-1].content or "").endswith("current")
    assert manager.calls == [("cli:chat", 20)]
