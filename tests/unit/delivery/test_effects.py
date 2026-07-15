from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.delivery.effects import EffectRepository, EffectRequest, EffectTransition
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import (
    InboundCommand,
    InterruptCommand,
    LostLeaseError,
    OperationalRepository,
    RunClaim,
)

NOW = datetime(2026, 7, 14, 11, 0, tzinfo=UTC)


def _claimed(
    tmp_path: Path,
) -> tuple[OperationalRepository, EffectRepository, SessionLease, RunClaim, int]:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    operational = OperationalRepository(database)
    accepted = operational.accept_inbound(
        InboundCommand(
            event_id="event-1",
            message_id="message-1",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "你好", "chat_id": "chat-1"},
            received_at=NOW,
        )
    )
    epoch = operational.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="lease",
        redis_value=f"worker-a|epoch|{epoch}",
    )
    claim = operational.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    return operational, EffectRepository(database), lease, claim, accepted.activity_version


def _request(
    claim: RunClaim,
    lease: SessionLease,
    activity_version: int,
    *,
    text: str = "回复",
    cancel_on_activity: bool = True,
) -> EffectRequest:
    return EffectRequest(
        operation_id=f"{claim.run_id}:final-text",
        run_id=claim.run_id,
        session_key=claim.session_key,
        channel="feishu",
        chat_id="chat-1",
        text=text,
        expected_activity_version=activity_version,
        lease=lease,
        now=NOW,
        cancel_on_activity=cancel_on_activity,
    )


def test_effect_identity_is_stable_and_same_operation_cannot_change_payload(
    tmp_path: Path,
) -> None:
    _, effects, lease, claim, activity_version = _claimed(tmp_path)

    first = effects.create(_request(claim, lease, activity_version))
    duplicate = effects.create(_request(claim, lease, activity_version))

    assert duplicate == first
    assert len(first.provider_uuid) == 36
    assert first.payload_hash
    with pytest.raises(ValueError, match="payload"):
        effects.create(_request(claim, lease, activity_version, text="另一条回复"))


def test_begin_send_atomically_checks_fence_and_activity_version(tmp_path: Path) -> None:
    operational, effects, lease, claim, activity_version = _claimed(tmp_path)
    effect = effects.create(_request(claim, lease, activity_version))
    operational.accept_inbound(
        InboundCommand(
            event_id="event-2",
            message_id="message-2",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "新消息"},
            received_at=NOW + timedelta(seconds=1),
        )
    )

    transition = effects.begin_send(effect.operation_id, lease=lease, now=NOW)

    assert transition is EffectTransition.CANCELLED
    assert effects.get(effect.operation_id).state == "cancelled"  # type: ignore[union-attr]

    new_epoch = operational.allocate_fence(
        "feishu:chat-1",
        owner_id="worker-b",
        now=NOW + timedelta(seconds=2),
    )
    assert new_epoch > lease.epoch
    with pytest.raises(LostLeaseError):
        effects.begin_send(effect.operation_id, lease=lease, now=NOW)


def test_passive_effect_is_not_cancelled_by_later_ordinary_message(tmp_path: Path) -> None:
    operational, effects, lease, claim, activity_version = _claimed(tmp_path)
    effect = effects.create(
        _request(claim, lease, activity_version, cancel_on_activity=False)
    )
    operational.accept_inbound(
        InboundCommand(
            event_id="event-2",
            message_id="message-2",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "排队的新问题"},
            received_at=NOW + timedelta(seconds=1),
        )
    )

    assert effects.begin_send(effect.operation_id, lease=lease, now=NOW) is EffectTransition.SEND


def test_targeted_stop_cancels_passive_effect_before_send(tmp_path: Path) -> None:
    operational, effects, lease, claim, activity_version = _claimed(tmp_path)
    effect = effects.create(
        _request(claim, lease, activity_version, cancel_on_activity=False)
    )
    operational.request_interrupt(
        InterruptCommand(
            event_id="stop-event",
            message_id="stop-message",
            session_key=claim.session_key,
            channel="feishu",
            chat_id="chat-1",
            requested_at=NOW,
        )
    )

    transition = effects.begin_send(effect.operation_id, lease=lease, now=NOW)

    assert transition is EffectTransition.CANCELLED


def test_unknown_effect_requires_explicit_reconciliation_and_expires_after_one_hour(
    tmp_path: Path,
) -> None:
    _, effects, lease, claim, activity_version = _claimed(tmp_path)
    effect = effects.create(_request(claim, lease, activity_version))
    assert effects.begin_send(effect.operation_id, lease=lease, now=NOW) is EffectTransition.SEND
    effects.mark_unknown(effect.operation_id, lease=lease, error="response lost", now=NOW)

    automatic_retry = effects.begin_send(
        effect.operation_id,
        lease=lease,
        now=NOW + timedelta(minutes=59),
    )
    assert automatic_retry is EffectTransition.NEEDS_REVIEW
    assert effects.get(effect.operation_id).state == "unknown"  # type: ignore[union-attr]
    assert effects.get(effect.operation_id).provider_uuid == effect.provider_uuid  # type: ignore[union-attr]

    explicit_retry = effects.begin_reconciliation(
        effect.operation_id,
        lease=lease,
        now=NOW + timedelta(minutes=59),
    )
    assert explicit_retry is EffectTransition.SEND
    effects.mark_unknown(
        effect.operation_id,
        lease=lease,
        error="response lost again",
        now=NOW + timedelta(minutes=59),
    )

    expired = effects.begin_reconciliation(
        effect.operation_id,
        lease=lease,
        now=NOW + timedelta(hours=1, seconds=1),
    )

    assert expired is EffectTransition.NEEDS_REVIEW
    assert effects.get(effect.operation_id).state == "needs_review"  # type: ignore[union-attr]


def test_effects_can_be_listed_for_operator_review(tmp_path: Path) -> None:
    _, effects, lease, claim, activity_version = _claimed(tmp_path)
    effect = effects.create(_request(claim, lease, activity_version))
    assert effects.begin_send(effect.operation_id, lease=lease, now=NOW) is EffectTransition.SEND
    effects.mark_unknown(effect.operation_id, lease=lease, error="lost", now=NOW)

    records = effects.list_reviewable()

    assert [record.operation_id for record in records] == [effect.operation_id]
