from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.operational import InboundCommand, OperationalRepository

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def _claimed_turn(tmp_path: Path) -> tuple[OperationalRepository, object, object]:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    accepted = repository.accept_inbound(
        InboundCommand(
            event_id="event-1",
            message_id="message-1",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "你好"},
            received_at=NOW,
        )
    )
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-1", now=NOW)
    lease = SimpleNamespace(
        session_key="feishu:chat-1",
        owner_id="worker-1",
        epoch=epoch,
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    return repository, claim, lease


def test_successful_turn_atomically_saves_messages_and_enqueues_consolidation(
    tmp_path: Path,
) -> None:
    repository, claim, lease = _claimed_turn(tmp_path)

    result = repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content="你好",
        assistant_content="你好，有什么可以帮你？",
        now=NOW,
    )

    assert result.created is True
    assert repository.get_job(claim.job_id).state == "succeeded"  # type: ignore[union-attr]
    memory_job = repository.get_job(result.consolidation_job_id)
    assert memory_job is not None
    assert (memory_job.kind, memory_job.priority, memory_job.state) == (
        "memory.consolidate",
        3,
        "queued",
    )
    assert repository.count("messages") == 2
    assert repository.count("outbox_events") == 2
    assert [(item.role, item.content) for item in repository.list_recent_messages(
        "feishu:chat-1", limit=10
    )] == [
        ("user", "你好"),
        ("assistant", "你好，有什么可以帮你？"),
    ]


def test_successful_turn_commit_is_idempotent(tmp_path: Path) -> None:
    repository, claim, lease = _claimed_turn(tmp_path)
    first = repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content="你好",
        assistant_content="回复",
        now=NOW,
    )

    second = repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content="你好",
        assistant_content="回复",
        now=NOW,
    )

    assert second == type(second)(first.consolidation_job_id, first.outbox_id, False)
    assert repository.count("messages") == 2
    assert repository.count("agent_jobs") == 2
    assert repository.count("outbox_events") == 2


def test_failure_before_turn_commit_rolls_back_messages_and_terminal_state(
    tmp_path: Path,
) -> None:
    repository, claim, lease = _claimed_turn(tmp_path)

    def failpoint(name: str) -> None:
        if name == "before_commit":
            raise RuntimeError("模拟 Turn commit 前崩溃")

    with pytest.raises(RuntimeError, match="commit 前"):
        repository.commit_successful_turn(
            claim.run_id,
            lease=lease,
            user_content="你好",
            assistant_content="回复",
            now=NOW,
            failpoint=failpoint,
        )

    assert repository.count("messages") == 0
    assert repository.count("agent_jobs") == 1
    assert repository.get_job(claim.job_id).state == "running"  # type: ignore[union-attr]


def test_recent_messages_applies_sliding_window_in_chronological_order(
    tmp_path: Path,
) -> None:
    repository, claim, lease = _claimed_turn(tmp_path)
    repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content="第一问",
        assistant_content="第一答",
        now=NOW,
    )
    accepted = repository.accept_inbound(
        InboundCommand(
            event_id="event-2",
            message_id="message-2",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "第二问"},
            received_at=NOW,
        )
    )
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-2", now=NOW)
    second_lease = SimpleNamespace(
        session_key="feishu:chat-1", owner_id="worker-2", epoch=epoch
    )
    second_claim = repository.claim_job(accepted.job_id, lease=second_lease, now=NOW)
    assert second_claim is not None
    repository.commit_successful_turn(
        second_claim.run_id,
        lease=second_lease,
        user_content="第二问",
        assistant_content="第二答",
        now=NOW,
    )

    recent = repository.list_recent_messages("feishu:chat-1", limit=3)

    assert [(item.role, item.content) for item in recent] == [
        ("assistant", "第一答"),
        ("user", "第二问"),
        ("assistant", "第二答"),
    ]
