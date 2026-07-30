from __future__ import annotations

from collections.abc import Sequence

import pytest

from memopilot.runtime.contracts import ChatMessage, ModelResponse
from memopilot.runtime.engine import AgentRuntime, SessionHistoryRequest, TurnInput
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

    async def complete(
        self, *, messages: Sequence[ChatMessage], tools: object
    ) -> ModelResponse:
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
