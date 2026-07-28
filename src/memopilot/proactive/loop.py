"""主动决策与渠道发送之间的轻量运行循环。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol

from memopilot.proactive.service import ProactiveOutcome
from memopilot.runtime.outbound import DeliveryError, OutboundDispatch, OutboundPort
from memopilot.tasks.lease import SessionLease


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

ProactiveServiceFactory = Callable[
    [str, int, SessionLease],
    ProactiveExecutionService,
]


class ProactiveLoop:
    def __init__(
        self,
        *,
        service_factory: ProactiveServiceFactory,
        outbound: OutboundPort,
    ) -> None:
        self._service_factory = service_factory
        self._outbound = outbound

    async def execute_task(
        self,
        *,
        task_id: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> str:
        chat_id = _required_text(payload, "chat_id")
        channel = _required_text(payload, "channel")
        activity_version = _integer(payload.get("activity_version"))
        service = self._service_factory(session_key, activity_version, lease)
        outcome = await service.execute(
            task_id=task_id,
            session_key=session_key,
            chat_id=chat_id,
            activity_version=activity_version,
            now=now,
        )
        if outcome.action == "drift":
            return "drift"
        if outcome.action != "send":
            return "succeeded"
        if outcome.decision_id is None:
            raise RuntimeError("Proactive send outcome 缺少 decision_id")
        sent = await self._outbound.dispatch(
            OutboundDispatch(channel=channel, chat_id=chat_id, content=outcome.message)
        )
        if sent:
            service.finalize_confirmed(outcome, confirmed_at=now)
            return "succeeded"
        raise DeliveryError("主动消息未明确发送成功")


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
