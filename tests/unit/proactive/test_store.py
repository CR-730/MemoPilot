from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.persistence.migrations import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
from memopilot.proactive.mcp_sources import (
    ProactiveEvent,
    ProactiveFetchResult,
    stable_ack_operation_id,
)
from memopilot.proactive.service import _decision_id
from memopilot.proactive.store import (
    ProactiveRepository,
    stable_proactive_effect_operation_id,
)

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _repository(tmp_path: Path) -> ProactiveRepository:
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    return ProactiveRepository(database)


def _event(event_id: str, kind: str = "content") -> ProactiveEvent:
    return ProactiveEvent(
        source_id="news",
        event_id=event_id,
        kind=kind,
        occurred_at=NOW.isoformat(),
        payload={"title": event_id},
    )


def test_fetch_commit_deduplicates_events_by_source_and_event_id(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    inserted = repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"), _event("e2", "alert"))),
        fetched_at=NOW,
    )
    duplicate = repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW + timedelta(minutes=1),
    )

    assert inserted == 2
    assert duplicate == 0
    assert [item.source_event_id for item in repository.list_unconsumed("feishu:chat-1")] == [
        "e2",
        "e1",
    ]


def test_consumed_content_reenters_after_ack_ttl_with_new_auditable_occurrence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    repository = ProactiveRepository(database)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )
    first = repository.list_unconsumed("feishu:chat-1")[0]
    decision = repository.create_decision(
        decision_id="first-e1",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="skip",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW,
        ack_ttl_hours={"news:e1": 24},
    )
    repository.commit_skip(decision.decision_id, committed_at=NOW)
    pending = repository.list_pending_acknowledgements(NOW)[0]
    repository.mark_acknowledged(pending.acknowledgement_id, acknowledged_at=NOW)

    before_expiry = repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW + timedelta(hours=23, minutes=59),
    )
    at_expiry = repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW + timedelta(hours=24),
    )

    assert before_expiry == 0
    assert at_expiry == 1
    second = repository.list_unconsumed("feishu:chat-1")[0]
    assert second.source_event_id == "e1"
    assert second.reservoir_id != first.reservoir_id
    assert _decision_id("feishu:chat-1", "content", (first,), 1) != _decision_id(
        "feishu:chat-1", "content", (second,), 1
    )
    with connect_database(database) as connection:
        archived = connection.execute(
            "SELECT reservoir_id, source_event_id, ack_ttl_hours FROM source_event_history"
        ).fetchall()
    assert [tuple(row) for row in archived] == [(first.reservoir_id, "e1", 24)]

    second_decision = repository.create_decision(
        decision_id=_decision_id("feishu:chat-1", "content", (second,), 1),
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="skip",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW + timedelta(hours=24),
        ack_ttl_hours={"news:e1": 24},
    )
    repository.commit_skip(second_decision.decision_id, committed_at=NOW + timedelta(hours=24))
    replayed_ack = repository.list_pending_acknowledgements(NOW + timedelta(hours=24))
    assert [(item.source_event_id, item.ttl_hours) for item in replayed_ack] == [("e1", 24)]


def test_alert_event_id_remains_permanently_deduplicated(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("a1", "alert"),)),
        fetched_at=NOW,
    )
    decision = repository.create_decision(
        decision_id="alert-a1",
        session_key="feishu:chat-1",
        trigger_kind="alert",
        action="alert",
        source_events=(("news", "a1"),),
        activity_version=1,
        decided_at=NOW,
    )
    repository.finalize_confirmed(
        decision.decision_id,
        is_effect_confirmed=lambda _operation_id: True,
        committed_at=NOW,
    )
    pending = repository.list_pending_acknowledgements(NOW)[0]
    repository.mark_acknowledged(pending.acknowledgement_id, acknowledged_at=NOW)

    inserted = repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("a1", "alert"),)),
        fetched_at=NOW + timedelta(days=30),
    )

    assert inserted == 0
    assert repository.list_unconsumed("feishu:chat-1") == ()


def test_alerts_are_read_newest_first_within_the_same_source(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    older = ProactiveEvent(
        source_id="news",
        event_id="older",
        kind="alert",
        occurred_at=(NOW - timedelta(minutes=1)).isoformat(),
        payload={"title": "旧告警"},
    )
    newer = ProactiveEvent(
        source_id="news",
        event_id="newer",
        kind="alert",
        occurred_at=NOW.isoformat(),
        payload={"title": "新告警"},
    )
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((older, newer)),
        fetched_at=NOW,
    )

    assert [item.source_event_id for item in repository.list_unconsumed("feishu:chat-1")] == [
        "newer",
        "older",
    ]


def test_proactive_audit_records_observation_and_drift(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.record_observation(
        session_key="feishu:chat-1",
        kind="content",
        subject_id="decision-1",
        trigger={"score": 0.9},
        candidates=({"item_id": "a"},),
        llm_input=({"role": "user", "content": "候选"},),
        created_at=NOW,
    )
    repository.mark_drift_started(
        session_key="feishu:chat-1",
        job_id="job-1",
        started_at=NOW,
    )

    assert repository.list_observations("feishu:chat-1")[0]["subject_id"] == "decision-1"
    drift = repository.list_drift_history("feishu:chat-1")[0]
    assert drift["job_id"] == "job-1"
    assert drift["outcome"] == "running"


def test_failed_event_serialization_does_not_change_existing_reservoir(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )
    invalid = ProactiveEvent(
        source_id="news",
        event_id="broken",
        kind="content",
        occurred_at=NOW.isoformat(),
        payload={"not-json": object()},
    )

    try:
        repository.commit_fetch(
            session_key="feishu:chat-1",
            source_id="news",
            result=ProactiveFetchResult((invalid,)),
            fetched_at=NOW,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("不可序列化 payload 应导致事务失败")

    assert [item.source_event_id for item in repository.list_unconsumed("feishu:chat-1")] == ["e1"]


def test_confirmed_finish_only_consumes_decision_events_and_queues_stable_ack(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"), _event("e2"), _event("e3"))),
        fetched_at=NOW,
    )
    decision = repository.create_decision(
        decision_id="decision-1",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="share",
        source_events=(("news", "e1"), ("news", "e2")),
        activity_version=4,
        reason="interesting",
        ack_ttl_hours={"news:e1": 168, "news:e2": 24},
        decided_at=NOW,
    )

    assert decision.effect_operation_id == stable_proactive_effect_operation_id("decision-1")
    assert repository.finalize_confirmed(
        "decision-1",
        is_effect_confirmed=lambda operation_id: operation_id == decision.effect_operation_id,
        committed_at=NOW + timedelta(seconds=1),
    )

    assert [item.source_event_id for item in repository.list_unconsumed("feishu:chat-1")] == ["e3"]
    pending = repository.list_pending_acknowledgements(NOW + timedelta(seconds=1))
    assert [(item.source_event_id, item.ack_operation_id, item.ttl_hours) for item in pending] == [
        ("e1", stable_ack_operation_id("news", "e1"), 168),
        ("e2", stable_ack_operation_id("news", "e2"), 24),
    ]


def test_unconfirmed_effect_does_not_consume_or_queue_ack(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )
    repository.create_decision(
        decision_id="decision-1",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="share",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW,
    )

    assert not repository.finalize_confirmed(
        "decision-1", is_effect_confirmed=lambda _: False, committed_at=NOW
    )
    assert len(repository.list_unconsumed("feishu:chat-1")) == 1
    assert repository.list_pending_acknowledgements(NOW) == ()


def test_recreating_same_decision_is_idempotent(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )
    arguments = {
        "decision_id": "decision-1",
        "session_key": "feishu:chat-1",
        "trigger_kind": "content",
        "action": "share",
        "source_events": (("news", "e1"),),
        "activity_version": 1,
        "decided_at": NOW,
    }

    first = repository.create_decision(**arguments)
    second = repository.create_decision(**arguments)

    assert second == first


def test_pending_decision_persists_complete_recovery_payload(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )
    created = repository.create_decision(
        decision_id="decision-recovery",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="share",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW,
        message="恢复后直接发送这条消息",
        evidence=("reservoir-e1",),
        reason="有可靠正文",
    )

    recovered = repository.find_pending_decision("feishu:chat-1")

    assert recovered == created
    assert recovered.message == "恢复后直接发送这条消息"
    assert recovered.evidence == ("reservoir-e1",)
    assert recovered.reason == "有可靠正文"


def test_decision_cannot_claim_another_sessions_event(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:other",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )

    try:
        repository.create_decision(
            decision_id="decision-1",
            session_key="feishu:chat-1",
            trigger_kind="content",
            action="skip",
            source_events=(("news", "e1"),),
            activity_version=1,
            decided_at=NOW,
        )
    except ValueError as exc:
        assert "会话" in str(exc)
    else:
        raise AssertionError("Decision 不应消费其他会话的事件")


def test_skip_consumes_only_selected_events_without_waiting_for_effect(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"), _event("e2"))),
        fetched_at=NOW,
    )
    repository.create_decision(
        decision_id="skip-1",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="skip",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW,
    )

    repository.commit_skip("skip-1", committed_at=NOW)

    assert [item.source_event_id for item in repository.list_unconsumed("feishu:chat-1")] == ["e2"]


def test_context_decision_consumes_without_creating_pending_ack(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("state-1", "context"),)),
        fetched_at=NOW,
    )
    repository.create_decision(
        decision_id="context-skip-1",
        session_key="feishu:chat-1",
        trigger_kind="context",
        action="skip_event",
        source_events=(("news", "state-1"),),
        activity_version=1,
        decided_at=NOW,
    )

    repository.commit_skip("context-skip-1", committed_at=NOW)

    assert repository.list_unconsumed("feishu:chat-1") == ()
    assert repository.list_pending_acknowledgements(NOW) == ()


def test_share_decision_cannot_bypass_effect_confirmation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )
    repository.create_decision(
        decision_id="share-1",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="share",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW,
    )

    try:
        repository.commit_skip("share-1", committed_at=NOW)
    except ValueError as exc:
        assert "share" in str(exc)
    else:
        raise AssertionError("share 决策不能绕过 Effect confirmed")

    assert len(repository.list_unconsumed("feishu:chat-1")) == 1


def test_decision_requires_non_empty_source_events(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(ValueError, match="至少引用一个"):
        repository.create_decision(
            decision_id="empty",
            session_key="feishu:chat-1",
            trigger_kind="content",
            action="skip",
            source_events=(),
            activity_version=1,
            decided_at=NOW,
        )


def test_decision_rejects_event_of_another_kind(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("alert-1", "alert"),)),
        fetched_at=NOW,
    )

    with pytest.raises(ValueError, match="类型"):
        repository.create_decision(
            decision_id="wrong-kind",
            session_key="feishu:chat-1",
            trigger_kind="content",
            action="skip",
            source_events=(("news", "alert-1"),),
            activity_version=1,
            decided_at=NOW,
        )


def test_consumed_event_cannot_be_reused_by_a_new_decision(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.commit_fetch(
        session_key="feishu:chat-1",
        source_id="news",
        result=ProactiveFetchResult((_event("e1"),)),
        fetched_at=NOW,
    )
    repository.create_decision(
        decision_id="first",
        session_key="feishu:chat-1",
        trigger_kind="content",
        action="skip",
        source_events=(("news", "e1"),),
        activity_version=1,
        decided_at=NOW,
    )
    repository.commit_skip("first", committed_at=NOW)

    with pytest.raises(ValueError, match="已消费"):
        repository.create_decision(
            decision_id="second",
            session_key="feishu:chat-1",
            trigger_kind="content",
            action="share",
            source_events=(("news", "e1"),),
            activity_version=1,
            decided_at=NOW,
        )


def test_context_transition_and_global_throttle_are_persisted(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    assert repository.save_context(
        source_id="presence",
        payload={"presence": "active"},
        fingerprint="active",
        observed_at=NOW,
    )
    assert not repository.save_context(
        source_id="presence",
        payload={"presence": "active"},
        fingerprint="active",
        observed_at=NOW + timedelta(minutes=1),
    )
    stored_context = repository.load_context("presence")
    assert stored_context is not None
    assert stored_context.payload == {"presence": "active"}
    assert [item.source_id for item in repository.list_contexts()] == ["presence"]
    assert repository.claim_context_reevaluation(NOW, min_interval=timedelta(hours=3))
    assert not repository.claim_context_reevaluation(
        NOW + timedelta(hours=2), min_interval=timedelta(hours=3)
    )
    state = repository.load_context_reevaluation_state()
    assert state == {
        "last_signaled_at": NOW.isoformat(),
        "last_candidate_at": (NOW + timedelta(hours=2)).isoformat(),
        "suppressed_count": 1,
    }
    assert repository.claim_context_reevaluation(
        NOW + timedelta(hours=3), min_interval=timedelta(hours=3)
    )


def test_drift_progress_survives_repository_restart(tmp_path: Path) -> None:
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    repository = ProactiveRepository(database)
    repository.mark_drift_started(
        session_key="feishu:chat-1",
        job_id="job-1",
        started_at=NOW,
    )

    reopened = ProactiveRepository(database)

    assert reopened.load_last_drift_at("feishu:chat-1") == NOW
