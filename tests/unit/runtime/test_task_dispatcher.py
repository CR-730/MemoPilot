from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.lease import SessionLease

NOW = datetime(2026, 7, 30, tzinfo=UTC)
LEASE = SessionLease("cli:chat", "agent", 1, "key", "value")


class _Handler:
    def __init__(self) -> None:
        self.calls: list[tuple[AgentTask, SessionLease]] = []

    async def execute_task(
        self, task: AgentTask, *, lease: SessionLease, now: datetime
    ) -> tuple[AgentTask, ...]:
        del now
        self.calls.append((task, lease))
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

    handlers = {name: _Handler() for name in ("passive", "memory", "proactive", "scheduler")}
    dispatcher = TaskDispatcher(**handlers)
    task = AgentTask("t1", kind, 0, "cli:chat", {"content": "hi"}, NOW)

    assert await dispatcher.dispatch(task, lease=LEASE, now=NOW) == ()
    assert handlers[attribute].calls == [(task, LEASE)]
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
        await dispatcher.dispatch(
            AgentTask("t1", "other", 0, "cli:chat", {}, NOW), lease=LEASE, now=NOW
        )
