"""Redis 后台任务到领域处理器的轻量执行边界。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Protocol

from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.outbound import OutboundDispatch, OutboundPort
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.redis_queue import QueueMessage


class ProactiveTaskExecutor(Protocol):
    async def execute_task(
        self,
        *,
        task_id: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> str: ...


class MemoryTaskExecutor(Protocol):
    async def execute(
        self,
        *,
        kind: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
    ) -> None: ...


class DriftTaskExecutor(Protocol):
    async def execute_task(
        self,
        *,
        task_id: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> object: ...


class BackgroundTaskDispatcher:
    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        repository: OperationalRepository,
        outbound: OutboundPort,
        memory_tasks: MemoryTaskExecutor,
        proactive: ProactiveTaskExecutor,
        drift: DriftTaskExecutor,
    ) -> None:
        self.runtime = runtime
        self.repository = repository
        self.outbound = outbound
        self.memory_tasks = memory_tasks
        self.proactive = proactive
        self.drift = drift

    async def execute(
        self,
        message: QueueMessage,
        *,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> str:
        if message.kind.startswith("memory."):
            await self.memory_tasks.execute(
                kind=message.kind,
                session_key=message.session_key,
                payload=payload,
                lease=lease,
            )
            return "succeeded"
        if message.kind == "proactive.tick":
            outcome = await self.proactive.execute_task(
                task_id=message.task_id,
                session_key=message.session_key,
                payload=payload,
                lease=lease,
                now=now,
            )
            if outcome != "drift":
                return outcome
            result = await self.drift.execute_task(
                task_id=message.task_id,
                session_key=message.session_key,
                payload=payload,
                lease=lease,
                now=now,
            )
            return str(getattr(result, "outcome", result))
        if message.kind == "drift.run":
            result = await self.drift.execute_task(
                task_id=message.task_id,
                session_key=message.session_key,
                payload=payload,
                lease=lease,
                now=now,
            )
            return str(getattr(result, "outcome", result))
        if message.kind == "schedule.run":
            try:
                return await self._run_schedule(message, payload, lease=lease, now=now)
            except BaseException as exc:
                if not isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    execution_id = str(payload.get("execution_id") or "")
                    if execution_id:
                        self.repository.transition_background_schedule(
                            execution_id,
                            lease=lease,
                            outcome="failed",
                            now=now,
                        )
                raise
        raise ValueError(f"不支持的后台任务: {message.kind}")

    async def _run_schedule(
        self,
        message: QueueMessage,
        payload: Mapping[str, object],
        *,
        lease: SessionLease,
        now: datetime,
    ) -> str:
        execution_id = _required_text(payload, "execution_id")
        current = self.repository.transition_background_schedule(
            execution_id,
            lease=lease,
            outcome="running",
            now=now,
        )
        if current in {"succeeded", "failed", "cancelled"}:
            return current
        task_payload = payload.get("payload")
        if not isinstance(task_payload, Mapping):
            raise ValueError("schedule.run payload.payload 必须是对象")
        mode = _required_text(payload, "execution_mode")
        if mode == "instant":
            text = _required_text(task_payload, "message")
        elif mode == "agent":
            result = await self.runtime.run(
                replace(
                    TurnInput(
                        session_key=message.session_key,
                        content=_required_text(task_payload, "prompt"),
                        prompt_scope="scheduled",
                        received_at=now,
                    ),
                    allowed_tool_risks=frozenset({"read-only", "write"}),
                    memory_source_ref=f"task:{message.task_id}",
                ),
            )
            if result.react.infrastructure_error:
                self.repository.transition_background_schedule(
                    execution_id, lease=lease, outcome="failed", now=now
                )
                return "failed"
            text = result.reply
        else:
            raise ValueError(f"未知 schedule execution_mode: {mode}")
        sent = await self.outbound.dispatch(
            OutboundDispatch(
                channel=_required_text(payload, "channel"),
                chat_id=_required_text(payload, "chat_id"),
                content=text,
            )
        )
        outcome = "succeeded" if sent else "failed"
        self.repository.transition_background_schedule(
            execution_id, lease=lease, outcome=outcome, now=now
        )
        return outcome


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"后台任务缺少 {key}")
    return value


__all__ = [
    "BackgroundTaskDispatcher",
    "DriftTaskExecutor",
    "MemoryTaskExecutor",
    "ProactiveTaskExecutor",
]
