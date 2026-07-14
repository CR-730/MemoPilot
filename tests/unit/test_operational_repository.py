from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.operational import (
    InboundCommand,
    InterruptCommand,
    OperationalRepository,
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
