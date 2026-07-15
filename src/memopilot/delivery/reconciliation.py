"""unknown/needs_review 外发副作用的本地人工处置服务。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from memopilot.delivery.effects import EffectRecord, EffectRepository
from memopilot.delivery.feishu import DeliveryResult, FinalResponseDispatcher
from memopilot.tasks.lease import SessionLease, SessionLeaseManager
from memopilot.tasks.operational import OperationalRepository


class EffectLeaseUnavailable(RuntimeError):
    """目标会话正在执行其他任务，人工处置暂不能取得 Lease。"""


class EffectReconciliationService:
    def __init__(
        self,
        operational: OperationalRepository,
        effects: EffectRepository,
        leases: SessionLeaseManager,
        dispatcher: FinalResponseDispatcher,
        *,
        owner_id: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._operational = operational
        self._effects = effects
        self._leases = leases
        self._dispatcher = dispatcher
        self._owner_id = owner_id
        self._clock = clock or (lambda: datetime.now(UTC))

    def list_reviewable(self) -> tuple[EffectRecord, ...]:
        return self._effects.list_reviewable()

    def show(self, operation_id: str) -> EffectRecord:
        return self._require_effect(operation_id)

    async def confirm(self, operation_id: str, *, message_id: str) -> EffectRecord:
        effect = self._require_effect(operation_id)
        lease = await self._acquire(effect)
        try:
            self._operational.resolve_effect_review(
                operation_id,
                lease=lease,
                decision="confirmed",
                message_id=message_id,
                now=self._clock(),
            )
            return self._require_effect(operation_id)
        finally:
            await self._leases.release(lease)

    async def fail(self, operation_id: str) -> EffectRecord:
        effect = self._require_effect(operation_id)
        lease = await self._acquire(effect)
        try:
            self._operational.resolve_effect_review(
                operation_id,
                lease=lease,
                decision="failed",
                message_id=None,
                now=self._clock(),
            )
            return self._require_effect(operation_id)
        finally:
            await self._leases.release(lease)

    async def retry(self, operation_id: str) -> DeliveryResult:
        effect = self._require_effect(operation_id)
        lease = await self._acquire(effect)
        try:
            return await self._dispatcher.retry_unknown(operation_id, lease=lease)
        finally:
            await self._leases.release(lease)

    async def _acquire(self, effect: EffectRecord) -> SessionLease:
        lease = await self._leases.acquire(
            effect.session_key,
            owner_id=self._owner_id,
            now=self._clock(),
        )
        if lease is None:
            raise EffectLeaseUnavailable(
                f"会话 {effect.session_key} 的 Lease 正忙，请稍后重试"
            )
        return lease

    def _require_effect(self, operation_id: str) -> EffectRecord:
        effect = self._effects.get(operation_id)
        if effect is None:
            raise KeyError(operation_id)
        return effect


__all__ = ["EffectLeaseUnavailable", "EffectReconciliationService"]
