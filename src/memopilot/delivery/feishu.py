"""Agent 最终文本到飞书 transport 的可靠发送服务。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, cast
from uuid import NAMESPACE_URL, uuid5

import httpx

from memopilot.channels.contracts import SendReceipt
from memopilot.channels.feishu import FeishuApiError
from memopilot.delivery.effects import (
    EffectRecord,
    EffectRepository,
    EffectRequest,
    EffectTransition,
)
from memopilot.delivery.feishu_live import FeishuLiveProgress, LiveCardTransport
from memopilot.tasks.operational import (
    FenceToken,
    OperationalRepository,
    RunClaim,
)

logger = logging.getLogger(__name__)


class TextTransport(Protocol):
    async def send(
        self,
        chat_id: str,
        message: str,
        *,
        provider_uuid: str,
    ) -> SendReceipt: ...


class DeliveryOutcome(StrEnum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    outcome: DeliveryOutcome
    effect: EffectRecord


class FinalResponseDispatcher:
    def __init__(
        self,
        operational: OperationalRepository,
        effects: EffectRepository,
        transport: TextTransport,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._operational = operational
        self._effects = effects
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(UTC))

    def create_live_progress(
        self,
        *,
        claim: RunClaim,
        lease: FenceToken,
    ) -> FeishuLiveProgress | None:
        try:
            if not hasattr(self._transport, "send_card") or not hasattr(
                self._transport, "patch_card"
            ):
                return None
            job = self._operational.get_job(claim.job_id)
            if job is None:
                return None
            chat_id = str(json.loads(job.payload_json).get("chat_id") or "")
            if not chat_id:
                return None
            provider_uuid = str(uuid5(NAMESPACE_URL, f"feishu:{claim.run_id}:live-card"))
            cancel_on_activity = job.kind != "agent.turn"
            return FeishuLiveProgress(
                cast(LiveCardTransport, cast(Any, self._transport)),
                chat_id=chat_id,
                provider_uuid=provider_uuid,
                authorize=lambda creating: self._operational.authorize_live_progress(
                    claim.run_id,
                    lease=lease,
                    expected_activity_version=job.activity_version,
                    now=self._clock(),
                    creating=creating,
                    cancel_on_activity=cancel_on_activity,
                ),
            )
        except Exception as exc:
            logger.warning("飞书 live 初始化失败，已降级为仅发送最终回复: %s", exc)
            return None

    async def dispatch(
        self,
        *,
        claim: RunClaim,
        lease: FenceToken,
        text: str,
        operation_id: str | None = None,
    ) -> DeliveryResult:
        job = self._operational.get_job(claim.job_id)
        if job is None:
            raise KeyError(claim.job_id)
        payload = json.loads(job.payload_json)
        chat_id = str(payload.get("chat_id") or "")
        if not chat_id:
            raise ValueError("Agent Job 缺少飞书 chat_id")
        now = self._clock()
        cancel_on_activity = job.kind != "agent.turn"
        effect = self._effects.create(
            EffectRequest(
                operation_id=operation_id or f"{claim.run_id}:final-text",
                run_id=claim.run_id,
                session_key=claim.session_key,
                channel="feishu",
                chat_id=chat_id,
                text=text,
                expected_activity_version=job.activity_version,
                lease=lease,
                now=now,
                cancel_on_activity=cancel_on_activity,
            )
        )
        transition = self._effects.begin_send(effect.operation_id, lease=lease, now=now)
        if transition is EffectTransition.CONFIRMED:
            return DeliveryResult(
                DeliveryOutcome.CONFIRMED,
                self._require_effect(effect.operation_id),
            )
        if transition is EffectTransition.CANCELLED:
            return DeliveryResult(
                DeliveryOutcome.CANCELLED,
                self._require_effect(effect.operation_id),
            )
        if transition is EffectTransition.NEEDS_REVIEW:
            return DeliveryResult(
                DeliveryOutcome.NEEDS_REVIEW,
                self._require_effect(effect.operation_id),
            )
        try:
            receipt = await self._transport.send(
                effect.chat_id,
                effect.text,
                provider_uuid=effect.provider_uuid,
            )
        except asyncio.CancelledError:
            self._effects.mark_unknown(
                effect.operation_id,
                lease=lease,
                error="send cancelled after request started; remote effect unknown",
                now=self._clock(),
            )
            raise
        except httpx.RequestError as exc:
            self._effects.mark_unknown(
                effect.operation_id,
                lease=lease,
                error=str(exc),
                now=self._clock(),
            )
            return DeliveryResult(
                DeliveryOutcome.NEEDS_REVIEW,
                self._require_effect(effect.operation_id),
            )
        except FeishuApiError as exc:
            self._effects.mark_known_failure(
                effect.operation_id,
                lease=lease,
                error=str(exc),
                now=self._clock(),
            )
            return DeliveryResult(
                DeliveryOutcome.FAILED,
                self._require_effect(effect.operation_id),
            )
        except Exception as exc:
            self._effects.mark_unknown(
                effect.operation_id,
                lease=lease,
                error=str(exc),
                now=self._clock(),
            )
            return DeliveryResult(
                DeliveryOutcome.NEEDS_REVIEW,
                self._require_effect(effect.operation_id),
            )
        self._effects.mark_confirmed(
            effect.operation_id,
            lease=lease,
            message_id=receipt.message_id,
            now=self._clock(),
        )
        return DeliveryResult(
            DeliveryOutcome.CONFIRMED,
            self._require_effect(effect.operation_id),
        )

    async def retry_unknown(
        self,
        operation_id: str,
        *,
        lease: FenceToken,
    ) -> DeliveryResult:
        """人工核对远端未成功后，显式重试不明确的飞书请求。"""
        now = self._clock()
        transition = self._effects.begin_reconciliation(
            operation_id,
            lease=lease,
            now=now,
        )
        effect = self._require_effect(operation_id)
        if transition is EffectTransition.CONFIRMED:
            self._operational.resolve_needs_review(
                effect.run_id,
                lease=lease,
                outcome="succeeded",
                now=now,
            )
            return DeliveryResult(DeliveryOutcome.CONFIRMED, effect)
        if transition is EffectTransition.NEEDS_REVIEW:
            return DeliveryResult(DeliveryOutcome.NEEDS_REVIEW, effect)
        if transition is EffectTransition.CANCELLED:
            self._operational.resolve_needs_review(
                effect.run_id,
                lease=lease,
                outcome="cancelled",
                now=now,
            )
            return DeliveryResult(
                DeliveryOutcome.CANCELLED,
                self._require_effect(operation_id),
            )
        try:
            receipt = await self._transport.send(
                effect.chat_id,
                effect.text,
                provider_uuid=effect.provider_uuid,
            )
        except asyncio.CancelledError:
            self._effects.mark_unknown(
                operation_id,
                lease=lease,
                error="reconciliation send cancelled after request started; remote effect unknown",
                now=self._clock(),
            )
            raise
        except httpx.RequestError as exc:
            self._effects.mark_unknown(
                operation_id,
                lease=lease,
                error=str(exc),
                now=self._clock(),
            )
            return DeliveryResult(
                DeliveryOutcome.NEEDS_REVIEW,
                self._require_effect(operation_id),
            )
        except FeishuApiError as exc:
            self._effects.mark_known_failure(
                operation_id,
                lease=lease,
                error=str(exc),
                now=self._clock(),
            )
            self._operational.resolve_needs_review(
                effect.run_id,
                lease=lease,
                outcome="failed",
                now=self._clock(),
            )
            return DeliveryResult(
                DeliveryOutcome.FAILED,
                self._require_effect(operation_id),
            )
        except Exception as exc:
            self._effects.mark_unknown(
                operation_id,
                lease=lease,
                error=str(exc),
                now=self._clock(),
            )
            return DeliveryResult(
                DeliveryOutcome.NEEDS_REVIEW,
                self._require_effect(operation_id),
            )
        finished_at = self._clock()
        self._effects.mark_confirmed(
            operation_id,
            lease=lease,
            message_id=receipt.message_id,
            now=finished_at,
        )
        self._operational.resolve_needs_review(
            effect.run_id,
            lease=lease,
            outcome="succeeded",
            now=finished_at,
        )
        return DeliveryResult(
            DeliveryOutcome.CONFIRMED,
            self._require_effect(operation_id),
        )

    def _require_effect(self, operation_id: str) -> EffectRecord:
        effect = self._effects.get(operation_id)
        if effect is None:
            raise RuntimeError(f"outbound effect 丢失: {operation_id}")
        return effect


__all__ = ["DeliveryOutcome", "DeliveryResult", "FinalResponseDispatcher", "TextTransport"]
