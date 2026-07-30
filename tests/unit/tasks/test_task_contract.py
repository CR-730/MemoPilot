from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.persistence.migrations import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
from memopilot.tasks.background import BackgroundTask
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import LostLeaseError, OperationalRepository


def test_background_task_is_serializable_without_operational_job_fields() -> None:
    task = BackgroundTask(
        task_id="execution-1",
        kind="schedule.run",
        priority=1,
        session_key="feishu:chat-1",
        payload={"message": "提醒"},
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )

    assert task.payload_json == '{"message": "提醒"}'
    assert not hasattr(task, "run_id")


def test_commit_turn_rejects_stale_fence(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-1"},
    )
    repository.record_inbound_activity(message)
    epoch = repository.allocate_fence(message.session_key, owner_id="stale", now=message.timestamp)
    stale = SessionLease(message.session_key, "stale", epoch, "lease", "value")
    repository.allocate_fence(message.session_key, owner_id="current", now=message.timestamp)

    with pytest.raises(LostLeaseError):
        repository.commit_turn(message, assistant_content="旧回复", lease=stale)

    assert repository.count("messages") == 0


@pytest.mark.parametrize(
    "invalid_media",
    ("broken", "{}", '["ok.png", 1]', '[""]'),
)
def test_commit_turn_rejects_different_explicit_media_on_replay(
    tmp_path: Path,
    invalid_media: str,
) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-media"},
    )

    committed = repository.commit_turn(
        message,
        assistant_content="回复",
        assistant_media=("first.png",),
    )

    assert committed is not None
    assert committed.media == ("first.png",)
    assert repository.commit_turn(message) is not None
    with pytest.raises(ValueError, match="不同媒体"):
        repository.commit_turn(
            message,
            assistant_content="回复",
            assistant_media=("other.png",),
        )

    with connect_database(database) as connection:
        connection.execute(
            "UPDATE messages SET media_json = ? WHERE role = 'assistant'",
            (invalid_media,),
        )
        connection.commit()
    with pytest.raises(ValueError, match="media_json"):
        repository.commit_turn(message)
    with connect_database(database) as connection:
        rows = connection.execute(
            "SELECT role, media_json FROM messages ORDER BY turn_position"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("user", "[]"),
        ("assistant", invalid_media),
    ]


def test_commit_turn_persists_tool_chain_and_replay_keeps_it(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    message = InboundMessage(
        "feishu", "user", "chat-1", "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-tools"},
    )
    tool_chain = ({"text": "调用工具", "calls": [{"call_id": "call-1", "result": "ok"}]},)

    assert repository.commit_turn(
        message,
        assistant_content="回复",
        tool_chain=tool_chain,
    ) is not None
    assert repository.commit_turn(message) is not None
    records = repository.list_recent_messages(message.session_key, limit=2)

    assert records[0].tool_chain == ()
    assert records[1].tool_chain == tool_chain


def test_commit_turn_fails_explicitly_for_corrupt_tool_chain_json(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    message = InboundMessage(
        "feishu", "user", "chat-1", "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-corrupt-tools"},
    )
    repository.commit_turn(message, assistant_content="回复")
    with connect_database(database) as connection:
        connection.execute(
            "UPDATE messages SET tool_chain_json = 'broken' WHERE role = 'assistant'"
        )

    with pytest.raises(ValueError, match="tool_chain_json"):
        repository.commit_turn(message)
