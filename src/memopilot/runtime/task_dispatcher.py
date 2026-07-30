"""仅将任务分流给领域入口。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from memopilot.tasks.agent_task import AgentTask


class TaskHandler(Protocol):
    async def execute_task(self, task: AgentTask, *, now: datetime) -> Sequence[AgentTask]: ...


class TaskDispatcher:
    def __init__(
        self,
        *,
        passive: TaskHandler,
        memory: TaskHandler,
        proactive: TaskHandler,
        scheduler: TaskHandler,
    ) -> None:
        self._passive = passive
        self._memory = memory
        self._proactive = proactive
        self._scheduler = scheduler

    async def dispatch(self, task: AgentTask, *, now: datetime) -> Sequence[AgentTask]:
        if task.kind == "passive.turn":
            return await self._passive.execute_task(task, now=now)
        if task.kind.startswith("memory."):
            return await self._memory.execute_task(task, now=now)
        if task.kind in {"proactive.tick", "drift.run"}:
            return await self._proactive.execute_task(task, now=now)
        if task.kind == "schedule.run":
            return await self._scheduler.execute_task(task, now=now)
        raise ValueError(f"不支持的 Agent 任务: {task.kind}")


__all__ = ["TaskDispatcher"]
