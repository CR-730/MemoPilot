from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.tasks.agent_task import AgentTask


def _repository(tmp_path: Path) -> ConversationRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return ConversationRepository(database)


def test_background_task_is_serializable_without_operational_job_fields() -> None:
    task = AgentTask(
        "execution-1",
        "schedule.run",
        1,
        "feishu:chat-1",
        {"message": "提醒"},
        datetime(2026, 7, 27, tzinfo=UTC),
    )
    assert task.payload_json == '{"message": "提醒"}'
    assert not hasattr(task, "run_id")


@pytest.mark.parametrize("invalid_media", ("broken", "{}", '["ok.png", 1]', '[""]'))
def test_turn_replay_rejects_corrupt_media(tmp_path: Path, invalid_media: str) -> None:
    repository = _repository(tmp_path)
    message = InboundMessage(
        "feishu", "user", "chat-1", "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-media"},
    )
    assert repository.commit_turn(
        message, assistant_content="回复", assistant_media=("first.png",)
    ) is not None
    with connect_database(repository.database) as connection:
        connection.execute(
            "UPDATE messages SET media_json = ? WHERE role = 'assistant'", (invalid_media,)
        )
    with pytest.raises(ValueError, match="media_json"):
        repository.find_committed_turn(message)


def test_turn_replay_keeps_tool_chain(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    message = InboundMessage(
        "feishu", "user", "chat-1", "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-tools"},
    )
    tool_chain = ({"calls": [{"call_id": "call-1", "result": "ok"}]},)
    assert repository.commit_turn(
        message, assistant_content="回复", assistant_tool_chain=tool_chain
    ) is not None
    assert repository.find_committed_turn(message) is not None
    assert repository.list_recent_messages(message.session_key, limit=2)[1].tool_chain == tool_chain


def test_commit_turn_requires_assistant_content(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    message = InboundMessage(
        "feishu", "user", "chat-1", "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-complete"},
    )

    with pytest.raises(TypeError):
        repository.commit_turn(message)  # type: ignore[call-arg]


def test_find_committed_turn_returns_persisted_reply(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    message = InboundMessage(
        "feishu", "user", "chat-1", "你好",
        timestamp=datetime(2026, 7, 28, tzinfo=UTC),
        metadata={"message_id": "message-read"},
    )
    repository.commit_turn(message, assistant_content="回复")

    committed = repository.find_committed_turn(message)

    assert committed is not None
    assert committed.assistant_content == "回复"
    assert committed.inserted is False
