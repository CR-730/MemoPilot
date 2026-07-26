"""Proactive 领域决策到可靠外发的生产适配层。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol

from memopilot.delivery.effects import EffectRepository
from memopilot.delivery.feishu import DeliveryOutcome, FinalResponseDispatcher
from memopilot.proactive.service import ProactiveOutcome
from memopilot.runtime.engine import TurnInput
from memopilot.tasks.operational import FenceToken, OperationalRepository, RunClaim


class ProactiveExecutionService(Protocol):
    async def execute(
        self,
        job_id: str,
        session_key: str,
        chat_id: str,
        activity_version: int,
        now: datetime,
    ) -> ProactiveOutcome: ...

    def finalize_confirmed(
        self, outcome: ProactiveOutcome, *, confirmed_at: datetime | None = None
    ) -> bool: ...

    def finalize_failed(
        self, outcome: ProactiveOutcome, *, failed_at: datetime | None = None
    ) -> bool: ...


ProactiveServiceFactory = Callable[[RunClaim, FenceToken], ProactiveExecutionService]


class ProactiveJobHandlerService:
    """一次 P2 Job 的外层适配，不把系统 Job 伪装成用户 Turn。"""

    def __init__(
        self,
        *,
        operational: OperationalRepository,
        effects: EffectRepository,
        dispatcher: FinalResponseDispatcher,
        service_factory: ProactiveServiceFactory,
    ) -> None:
        self._operational = operational
        self._effects = effects
        self._dispatcher = dispatcher
        self._service_factory = service_factory

    async def execute(
        self,
        *,
        payload: dict[str, object],
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
    ) -> str:
        del turn
        chat_id = _required_text(payload, "chat_id")
        job = self._operational.get_job(claim.job_id)
        if job is None:
            raise KeyError(claim.job_id)
        service = self._service_factory(claim, lease)
        outcome = await service.execute(
            job_id=claim.job_id,
            session_key=claim.session_key,
            chat_id=chat_id,
            activity_version=job.activity_version,
            now=now,
        )
        if outcome.action == "drift":
            return "drift"
        if outcome.action != "send":
            return "succeeded"
        if outcome.decision_id is None or outcome.effect_operation_id is None:
            raise RuntimeError("Proactive send outcome 缺少稳定 decision/effect identity")

        existing = self._effects.get(outcome.effect_operation_id)
        if existing is not None and existing.run_id != claim.run_id:
            if existing.state == "pending" and self._effects.adopt_pending(
                outcome.effect_operation_id,
                run_id=claim.run_id,
                expected_activity_version=job.activity_version,
                lease=lease,
                now=now,
            ):
                existing = self._effects.get(outcome.effect_operation_id)
                if existing is None:
                    raise RuntimeError("接管后的 Proactive Effect 消失")
            elif existing.state == "confirmed":
                service.finalize_confirmed(outcome, confirmed_at=now)
                return "succeeded"
            elif existing.state == "cancelled":
                service.finalize_failed(outcome, failed_at=now)
                return "failed"
            else:
                # sending/unknown 可能已经到达远端，新 Tick 只能观察，不能盲目重发。
                return "succeeded"

        delivery = await self._dispatcher.dispatch(
            claim=claim,
            lease=lease,
            text=outcome.message,
            operation_id=outcome.effect_operation_id,
        )
        mapped = {
            DeliveryOutcome.CONFIRMED: "succeeded",
            DeliveryOutcome.CANCELLED: "cancelled",
            DeliveryOutcome.FAILED: "failed",
            DeliveryOutcome.NEEDS_REVIEW: "needs_review",
        }[delivery.outcome]
        if delivery.outcome is DeliveryOutcome.CONFIRMED:
            service.finalize_confirmed(outcome, confirmed_at=now)
        elif delivery.outcome is DeliveryOutcome.FAILED:
            service.finalize_failed(outcome, failed_at=now)
        return mapped

def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Proactive Job 缺少 {key}")
    return value


__all__ = [
    "ProactiveExecutionService",
    "ProactiveJobHandlerService",
    "ProactiveServiceFactory",
]
