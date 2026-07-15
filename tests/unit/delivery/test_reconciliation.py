from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from memopilot.delivery.effects import EffectRecord
from memopilot.delivery.reconciliation import (
    EffectLeaseUnavailable,
    EffectReconciliationService,
)

NOW = datetime(2026, 7, 15, tzinfo=UTC)


def _effect() -> EffectRecord:
    return EffectRecord(
        operation_id="operation-1",
        run_id="run-1",
        session_key="feishu:chat-1",
        channel="feishu",
        chat_id="chat-1",
        payload_json='{"text":"回复"}',
        payload_hash="hash",
        provider_uuid="uuid",
        expected_activity_version=1,
        state="unknown",
        owner_id="worker-a",
        fencing_epoch=1,
        message_id=None,
        first_requested_at=NOW.isoformat(),
        cancel_on_activity=False,
    )


@pytest.mark.asyncio
async def test_confirm_acquires_session_lease_and_releases_it() -> None:
    effect = _effect()
    effects = Mock()
    effects.get.return_value = effect
    leases = AsyncMock()
    lease = SimpleNamespace(session_key=effect.session_key)
    leases.acquire.return_value = lease
    operational = Mock()
    service = EffectReconciliationService(
        operational,
        effects,
        leases,
        AsyncMock(),
        owner_id="operator-1",
        clock=lambda: NOW,
    )

    result = await service.confirm(effect.operation_id, message_id="om-observed")

    assert result is effect
    leases.acquire.assert_awaited_once_with(
        effect.session_key, owner_id="operator-1", now=NOW
    )
    operational.resolve_effect_review.assert_called_once_with(
        effect.operation_id,
        lease=lease,
        decision="confirmed",
        message_id="om-observed",
        now=NOW,
    )
    leases.release.assert_awaited_once_with(lease)


@pytest.mark.asyncio
async def test_operator_refuses_to_mutate_when_session_lease_is_busy() -> None:
    effects = Mock()
    effects.get.return_value = _effect()
    leases = AsyncMock()
    leases.acquire.return_value = None
    service = EffectReconciliationService(
        Mock(),
        effects,
        leases,
        AsyncMock(),
        owner_id="operator-1",
        clock=lambda: NOW,
    )

    with pytest.raises(EffectLeaseUnavailable):
        await service.fail("operation-1")
