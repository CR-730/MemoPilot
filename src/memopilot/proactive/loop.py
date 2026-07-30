"""主动决策与渠道发送之间的轻量运行循环。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.proactive.service import ProactiveOutcome
from memopilot.runtime.outbound import DeliveryError, OutboundDispatch, OutboundPort
from memopilot.tasks.agent_task import AgentTask


class ProactiveExecutionService(Protocol):
    async def execute(
        self,
        task_id: str,
        session_key: str,
        chat_id: str,
        activity_version: int,
        now: datetime,
    ) -> ProactiveOutcome: ...

    def finalize_confirmed(
        self, outcome: ProactiveOutcome, *, confirmed_at: datetime | None = None
    ) -> bool: ...

    def assert_current(self) -> None: ...


class DriftTaskRunner(Protocol):
    async def execute_task(
        self,
        *,
        task_id: str,
        session_key: str,
        payload: dict[str, object],
        now: datetime,
    ) -> object: ...


ProactiveServiceFactory = Callable[
    [str, int],
    ProactiveExecutionService,
]


class ProactiveLoop:
    def __init__(
        self,
        *,
        service_factory: ProactiveServiceFactory,
        outbound: OutboundPort,
        drift: DriftTaskRunner,
    ) -> None:
        self._service_factory = service_factory
        self._outbound = outbound
        self._drift = drift

    async def execute_task(
        self,
        task: AgentTask,
        *,
        now: datetime,
    ) -> tuple[AgentTask, ...]:
        if task.kind == "drift.run":
            await self._execute_drift(task, now=now)
            return ()
        if task.kind != "proactive.tick":
            raise ValueError(f"不支持的主动任务: {task.kind}")
        return await self._execute_tick(task, now=now)

    async def _execute_tick(self, task: AgentTask, *, now: datetime) -> tuple[AgentTask, ...]:
        task_id, session_key, payload = task.task_id, task.session_key, task.payload
        chat_id = _required_text(payload, "chat_id")
        channel = _required_text(payload, "channel")
        activity_version = _integer(payload.get("activity_version"))
        service = self._service_factory(session_key, activity_version)
        outcome = await service.execute(
            task_id=task_id,
            session_key=session_key,
            chat_id=chat_id,
            activity_version=activity_version,
            now=now,
        )
        if outcome.action == "drift":
            await self._execute_drift(task, now=now)
            return ()
        if outcome.action != "send":
            return ()
        if outcome.decision_id is None:
            raise RuntimeError("Proactive send outcome 缺少 decision_id")
        service.assert_current()
        sent = await self._outbound.dispatch(
            OutboundDispatch(
                channel=channel,
                chat_id=chat_id,
                content=outcome.message,
                metadata={
                    "provider_uuid": str(
                        uuid5(NAMESPACE_URL, f"memopilot:proactive:{outcome.decision_id}")
                    )
                },
            )
        )
        if sent:
            service.finalize_confirmed(outcome, confirmed_at=now)
            return ()
        raise DeliveryError("主动消息未明确发送成功")

    async def _execute_drift(self, task: AgentTask, *, now: datetime) -> None:
        await self._drift.execute_task(
            task_id=task.task_id,
            session_key=task.session_key,
            payload=task.payload,
            now=now,
        )


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Proactive 任务缺少 {key}")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return 0


__all__ = ["ProactiveExecutionService", "ProactiveLoop", "ProactiveServiceFactory"]
