"""仅将任务分流给领域入口。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from memopilot.bus.events import InboundMessage
from memopilot.tasks.agent_task import AgentTask


class TaskHandler(Protocol):
    async def execute_task(self, task: AgentTask, *, now: datetime) -> Sequence[AgentTask]: ...


class PassiveHandler(Protocol):
    async def process(
        self, message: InboundMessage, key: str, *, dispatch_outbound: bool = True
    ) -> Sequence[AgentTask]: ...


class TaskDispatcher:
    def __init__(
        self,
        *,
        passive: PassiveHandler,
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
            message = inbound_message_from_task(task)
            return await self._passive.process(message, task.session_key)
        if task.kind.startswith("memory."):
            return await self._memory.execute_task(task, now=now)
        if task.kind in {"proactive.tick", "drift.run"}:
            return await self._proactive.execute_task(task, now=now)
        if task.kind == "schedule.run":
            return await self._scheduler.execute_task(task, now=now)
        raise ValueError(f"不支持的 Agent 任务: {task.kind}")


def inbound_message_from_task(task: AgentTask) -> InboundMessage:
    payload = task.payload
    timestamp = datetime.fromisoformat(str(payload.get("timestamp") or ""))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    metadata, media = payload.get("metadata"), payload.get("media")
    channel, sender, chat_id, content = (
        str(payload.get(key) or "").strip() for key in ("channel", "sender", "chat_id", "content")
    )
    if not all((channel, sender, chat_id, content)):
        raise ValueError("passive.turn missing inbound fields")
    message = InboundMessage(
        channel,
        sender,
        chat_id,
        content,
        timestamp=timestamp,
        media=[str(item) for item in media] if isinstance(media, list) else [],
        metadata={str(key): value for key, value in metadata.items()}
        if isinstance(metadata, dict)
        else {},
    )
    if message.session_key != task.session_key:
        raise ValueError("passive.turn session_key mismatch")
    return message


__all__ = ["TaskDispatcher", "inbound_message_from_task"]
