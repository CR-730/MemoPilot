from datetime import UTC, datetime

import pytest

from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.tasks.operational import OperationalRepository

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _repository(tmp_path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    with connect_database(database) as connection:
        now = NOW.isoformat()
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-1', 'feishu', 'chat-1', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO session_activity(session_key, activity_version, updated_at) "
            "VALUES ('feishu:chat-1', 7, ?)",
            (now,),
        )
    return repository


def test_generic_system_job_is_stable_and_idempotent(tmp_path) -> None:
    repository = _repository(tmp_path)

    first = repository.enqueue_system_job(
        kind="proactive.tick",
        priority=2,
        session_key="feishu:chat-1",
        idempotency_key="proactive:123",
        activity_version=7,
        payload={"bucket": 123},
        now=NOW,
    )
    second = repository.enqueue_system_job(
        kind="proactive.tick",
        priority=2,
        session_key="feishu:chat-1",
        idempotency_key="proactive:123",
        activity_version=7,
        payload={"bucket": 123},
        now=NOW,
    )

    assert first.created is True
    assert second.created is False
    assert first.job_id == second.job_id
    assert first.outbox_id == second.outbox_id
    assert repository.count("agent_jobs") == 1
    assert repository.count("outbox_events") == 1


def test_generic_system_job_rolls_back_job_and_outbox_together(tmp_path) -> None:
    repository = _repository(tmp_path)

    def crash(point: str) -> None:
        if point == "before_commit":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        repository.enqueue_system_job(
            kind="proactive.tick",
            priority=2,
            session_key="feishu:chat-1",
            idempotency_key="proactive:rollback",
            activity_version=7,
            payload={"bucket": 124},
            now=NOW,
            failpoint=crash,
        )

    assert repository.count("agent_jobs") == 0
    assert repository.count("outbox_events") == 0


def test_single_private_session_query_ignores_system_sessions(tmp_path) -> None:
    repository = _repository(tmp_path)
    with connect_database(repository.database) as connection:
        now = NOW.isoformat()
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('system:memory', 'system', 'memory', ?, ?)",
            (now, now),
        )

    target = repository.get_single_private_session()

    assert target is not None
    assert target.session_key == "feishu:chat-1"
    assert target.channel == "feishu"
    assert target.chat_id == "chat-1"
    assert target.activity_version == 7
