from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.tasks.agent_task import AgentTask

NOW = datetime(2026, 7, 30, tzinfo=UTC)


class _Handler:
    def __init__(self) -> None:
        self.calls: list[AgentTask] = []

    async def execute_task(
        self, task: AgentTask, *, now: datetime
    ) -> tuple[AgentTask, ...]:
        del now
        self.calls.append(task)
        return ()


class _PassiveHandler:
    def __init__(self) -> None:
        self.calls: list[InboundMessage] = []

    async def process(
        self, message: InboundMessage, key: str, *, dispatch_outbound: bool = True
    ) -> tuple[AgentTask, ...]:
        assert key == message.session_key
        assert dispatch_outbound is True
        self.calls.append(message)
        return ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, attribute",
    [
        ("passive.turn", "passive"),
        ("memory.optimize", "memory"),
        ("proactive.tick", "proactive"),
        ("drift.run", "proactive"),
        ("schedule.run", "scheduler"),
    ],
)
async def test_dispatcher_only_routes_task(kind: str, attribute: str) -> None:
    from memopilot.runtime.task_dispatcher import TaskDispatcher

    handlers = {name: _Handler() for name in ("memory", "proactive", "scheduler")}
    passive = _PassiveHandler()
    handlers["passive"] = passive  # type: ignore[assignment]
    dispatcher = TaskDispatcher(**handlers)
    task = AgentTask(
        "t1",
        kind,
        0,
        "cli:chat",
        {
            "channel": "cli",
            "sender": "user",
            "chat_id": "chat",
            "content": "hi",
            "timestamp": NOW.isoformat(),
        },
        NOW,
    )

    assert await dispatcher.dispatch(task, now=NOW) == ()
    if attribute == "passive":
        assert passive.calls[0].content == "hi"
    else:
        assert handlers[attribute].calls == [task]
    assert not hasattr(dispatcher, "repository")
    assert not hasattr(dispatcher, "runtime")
    assert not hasattr(dispatcher, "outbound")


@pytest.mark.asyncio
async def test_dispatcher_rejects_unknown_kind() -> None:
    from memopilot.runtime.task_dispatcher import TaskDispatcher

    handler = _Handler()
    dispatcher = TaskDispatcher(
        passive=handler, memory=handler, proactive=handler, scheduler=handler
    )
    with pytest.raises(ValueError, match="不支持的 Agent 任务"):
        await dispatcher.dispatch(AgentTask("t1", "other", 0, "cli:chat", {}, NOW), now=NOW)
