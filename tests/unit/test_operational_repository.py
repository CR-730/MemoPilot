from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.persistence.migrations import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import (
    InboundCommand,
    InterruptCommand,
    OperationalRepository,
    PendingInterruptError,
    TurnInterruptSnapshot,
)

NOW = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)


def make_repository(tmp_path: Path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


def make_command(**overrides: object) -> InboundCommand:
    values: dict[str, object] = {
        "event_id": "event-1",
        "message_id": "message-1",
        "session_key": "feishu:chat-1",
        "channel": "feishu",
        "chat_id": "chat-1",
        "payload": {"text": "你好"},
        "received_at": NOW,
    }
    values.update(overrides)
    return InboundCommand(**values)  # type: ignore[arg-type]


def test_inbound_transaction_creates_one_job_and_outbox(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    result = repository.accept_inbound(make_command())

    assert result.created is True
    assert result.activity_version == 1
    assert repository.count("inbound_events") == 1
    assert repository.count("agent_jobs") == 1
    assert repository.count("outbox_events") == 1
    job = repository.get_job(result.job_id)
    assert job is not None
    assert job.priority == 0
    assert job.state == "queued"


@pytest.mark.parametrize(
    ("overrides"),
    [
        {"payload": {"text": "重复事件"}},
        {"event_id": "event-2", "payload": {"text": "重复消息"}},
    ],
)
def test_duplicate_event_or_message_returns_existing_result_without_incrementing(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    repository = make_repository(tmp_path)
    first = repository.accept_inbound(make_command())

    duplicate = repository.accept_inbound(make_command(**overrides))

    assert duplicate.created is False
    assert duplicate.job_id == first.job_id
    assert duplicate.outbox_id == first.outbox_id
    assert duplicate.activity_version == 1
    assert repository.get_activity_version("feishu:chat-1") == 1
    assert repository.count("inbound_events") == 1


def test_failure_before_commit_rolls_back_all_inbound_writes(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    def failpoint(name: str) -> None:
        if name == "before_commit":
            raise RuntimeError("模拟 commit 前崩溃")

    with pytest.raises(RuntimeError, match="commit 前"):
        repository.accept_inbound(make_command(), failpoint=failpoint)

    assert repository.count("sessions") == 0
    assert repository.count("session_activity") == 0
    assert repository.count("inbound_events") == 0
    assert repository.count("agent_jobs") == 0
    assert repository.count("outbox_events") == 0


def test_concurrent_duplicate_inbound_is_serialized_by_sqlite(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: repository.accept_inbound(make_command()), range(2)))

    assert sorted(result.created for result in results) == [False, True]
    assert {result.job_id for result in results} == {results[0].job_id}
    assert repository.get_activity_version("feishu:chat-1") == 1
    assert repository.count("agent_jobs") == 1
    assert repository.count("outbox_events") == 1


def test_session_identities_are_upserted_and_rebuilt_by_channel(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    repository.accept_inbound(make_command())

    repository.remember_session_identities(
        session_key="feishu:chat-1",
        channel="feishu",
        chat_id="chat-1",
        identities={"open_id": "ou_1", "user_id": "u_1", "union_id": "on_1"},
        now=NOW,
    )
    repository.remember_session_identities(
        session_key="feishu:chat-1",
        channel="feishu",
        chat_id="chat-1",
        identities={"open_id": "ou_1", "user_id": "u_1"},
        now=NOW,
    )

    identities = repository.list_session_identities("feishu")
    assert {(item.identity_kind, item.identity_value, item.chat_id) for item in identities} == {
        ("open_id", "ou_1", "chat-1"),
        ("user_id", "u_1", "chat-1"),
        ("union_id", "on_1", "chat-1"),
    }
    assert repository.count("session_identities") == 3


def test_duplicate_interrupt_only_increments_activity_once_and_creates_no_job(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    repository.accept_inbound(make_command())
    command = InterruptCommand(
        event_id="stop-event-1",
        message_id="stop-message-1",
        session_key="feishu:chat-1",
        channel="feishu",
        chat_id="chat-1",
        requested_at=NOW,
    )

    first = repository.request_interrupt(command)
    duplicate = repository.request_interrupt(command)

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.activity_version == first.activity_version == 2
    assert repository.get_activity_version("feishu:chat-1") == 2
    assert repository.count("session_interrupts") == 1
    assert repository.count("agent_jobs") == 1


def test_interrupt_targets_only_the_run_active_when_command_arrives(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    accepted = repository.accept_inbound(make_command())
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="lease",
        redis_value="worker-a|epoch|1",
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None

    result = repository.request_interrupt(
        InterruptCommand(
            event_id="stop-event",
            message_id="stop-message",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            requested_at=NOW,
        )
    )

    assert result.target_run_id == claim.run_id
    assert repository.has_pending_interrupt(claim.run_id) is True


def test_confirmed_reply_commits_turn_and_acknowledges_late_interrupt(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    accepted = repository.accept_inbound(make_command())
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="lease",
        redis_value=f"worker-a|epoch|{epoch}",
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    repository.request_interrupt(
        InterruptCommand(
            event_id="late-stop-event",
            message_id="late-stop-message",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            requested_at=NOW + timedelta(seconds=1),
        )
    )

    committed = repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content="你好",
        assistant_content="你好呀",
        now=NOW + timedelta(seconds=2),
        acknowledge_pending_interrupt=True,
    )

    assert committed.created is True
    assert repository.get_job(accepted.job_id).state == "succeeded"  # type: ignore[union-attr]
    with connect_database(repository.database) as connection:
        interrupt = connection.execute(
            "SELECT state FROM session_interrupts WHERE target_run_id = ?",
            (claim.run_id,),
        ).fetchone()
        messages = connection.execute(
            "SELECT role, content FROM messages ORDER BY session_position"
        ).fetchall()
    assert interrupt is not None and interrupt["state"] == "acknowledged"
    assert [tuple(row) for row in messages] == [
        ("user", "你好"),
        ("assistant", "你好呀"),
    ]


def test_commit_persists_cited_ids_and_enqueues_idempotent_reinforcement(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    accepted = repository.accept_inbound(make_command())
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="lease",
        redis_value=f"worker-a|epoch|{epoch}",
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None

    repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content="我喜欢什么？",
        assistant_content="你喜欢中文回答。",
        cited_memory_ids=("m1", "m2", "m1"),
        now=NOW,
    )

    with connect_database(repository.database) as connection:
        citation = connection.execute(
            "SELECT observation_json FROM steps "
            "WHERE run_id = ? AND step_type = 'memory_citation'",
            (claim.run_id,),
        ).fetchone()
        job = connection.execute(
            "SELECT kind, payload_json FROM agent_jobs "
            "WHERE kind = 'memory.reinforce'",
        ).fetchone()
    assert citation is not None
    assert json.loads(citation["observation_json"])["cited_memory_ids"] == ["m1", "m2"]
    assert job is not None and job["kind"] == "memory.reinforce"
    assert json.loads(job["payload_json"])["item_ids"] == ["m1", "m2"]


def test_unsent_reply_rejects_late_interrupt_without_partial_memory_commit(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    accepted = repository.accept_inbound(make_command())
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="lease",
        redis_value=f"worker-a|epoch|{epoch}",
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    repository.request_interrupt(
        InterruptCommand(
            event_id="late-stop-event",
            message_id="late-stop-message",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            requested_at=NOW + timedelta(seconds=1),
        )
    )

    with pytest.raises(PendingInterruptError):
        repository.commit_successful_turn(
            claim.run_id,
            lease=lease,
            user_content="你好",
            assistant_content="尚未发出的回复",
            now=NOW + timedelta(seconds=2),
            reject_pending_interrupt=True,
        )

    assert repository.count("messages") == 0
    assert repository.count("consolidation_manifests") == 0
    assert repository.get_job(accepted.job_id).state == "running"  # type: ignore[union-attr]


def test_interrupt_does_not_target_background_memory_run(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    accepted = repository.accept_inbound(make_command())
    first_epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    first_lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=first_epoch,
        redis_key="lease",
        redis_value=f"worker-a|epoch|{first_epoch}",
    )
    turn_claim = repository.claim_job(accepted.job_id, lease=first_lease, now=NOW)
    assert turn_claim is not None
    committed = repository.commit_successful_turn(
        turn_claim.run_id,
        lease=first_lease,
        user_content="你好",
        assistant_content="你好呀",
        now=NOW,
    )
    second_epoch = repository.allocate_fence(
        "feishu:chat-1",
        owner_id="worker-b",
        now=NOW + timedelta(seconds=1),
    )
    second_lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-b",
        epoch=second_epoch,
        redis_key="lease",
        redis_value=f"worker-b|epoch|{second_epoch}",
    )
    memory_claim = repository.claim_job(
        committed.consolidation_job_id,
        lease=second_lease,
        now=NOW + timedelta(seconds=1),
    )
    assert memory_claim is not None

    result = repository.request_interrupt(
        InterruptCommand(
            event_id="stop-memory-event",
            message_id="stop-memory-message",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            requested_at=NOW + timedelta(seconds=2),
        )
    )

    assert result.target_run_id is None
    assert repository.has_pending_interrupt(memory_claim.run_id) is False


def test_interrupted_run_snapshot_is_reserved_once_and_expires(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    accepted = repository.accept_inbound(make_command())
    epoch = repository.allocate_fence("feishu:chat-1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="lease",
        redis_value="worker-a|epoch|1",
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    repository.request_interrupt(
        InterruptCommand(
            event_id="stop-event",
            message_id="stop-message",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            requested_at=NOW,
        )
    )
    repository.finish_interrupted_run(
        claim.run_id,
        lease=lease,
        snapshot=TurnInterruptSnapshot(
            original_message="查询天气并发邮件",
            partial_reply="已经查到天气",
            partial_thinking="下一步准备发邮件",
            tools_used=("weather",),
            tool_chain=({"tool": "weather", "status": "done"},),
        ),
        now=NOW,
        ttl=timedelta(minutes=30),
    )

    resume = repository.accept_inbound(
        make_command(
            event_id="resume-event",
            message_id="resume-message",
            payload={"text": "继续"},
            received_at=NOW + timedelta(minutes=1),
        )
    )
    reserved = repository.reserve_interrupt_snapshot(
        "feishu:chat-1", job_id=resume.job_id, now=NOW + timedelta(minutes=29)
    )
    assert reserved is not None
    assert reserved.partial_reply == "已经查到天气"
    assert repository.reserve_interrupt_snapshot(
        "feishu:chat-1", job_id="another-job", now=NOW + timedelta(minutes=29)
    ) is None
    repository.release_interrupt_snapshot(reserved.snapshot_id, job_id=resume.job_id)
    late = repository.accept_inbound(
        make_command(
            event_id="late-event",
            message_id="late-message",
            payload={"text": "太晚了"},
            received_at=NOW + timedelta(minutes=31),
        )
    )
    assert repository.reserve_interrupt_snapshot(
        "feishu:chat-1", job_id=late.job_id, now=NOW + timedelta(minutes=31)
    ) is None
