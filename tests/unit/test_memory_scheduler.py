import json
from datetime import UTC, datetime, timedelta

from memopilot.memory.scheduler import MemoryMaintenanceScheduler
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.tasks.operational import OperationalRepository

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def _repository(tmp_path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


def test_same_optimizer_time_bucket_enqueues_only_one_job(tmp_path) -> None:
    repository = _repository(tmp_path)
    scheduler = MemoryMaintenanceScheduler(
        repository,
        enabled=True,
        interval=timedelta(hours=18),
    )

    first = scheduler.tick(now=NOW)
    second = scheduler.tick(now=NOW + timedelta(minutes=5))

    assert first is not None and first.created is True
    assert second is not None and second.created is False
    assert repository.count("agent_jobs") == 1
    assert repository.count("outbox_events") == 1


def test_disabled_optimizer_scheduler_enqueues_nothing(tmp_path) -> None:
    repository = _repository(tmp_path)
    scheduler = MemoryMaintenanceScheduler(
        repository,
        enabled=False,
        interval=timedelta(hours=18),
    )

    assert scheduler.tick(now=NOW) is None
    assert repository.count("agent_jobs") == 0


def test_scheduler_requeues_failed_vectorization_with_bounded_audit_job(tmp_path) -> None:
    repository = _repository(tmp_path)
    with connect_database(repository.database) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 0)",
            ("feishu:chat-1", "feishu", "chat-1", NOW.isoformat(), NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO agent_jobs(job_id, kind, priority, session_key, idempotency_key, "
            "state, activity_version, payload_json, created_at, updated_at, finished_at) "
            "VALUES ('vector-0', 'memory.vectorize', 3, 'feishu:chat-1', 'vector-0', "
            "'failed', 0, ?, ?, ?, ?)",
            (
                json.dumps({"consolidation_id": "con-1"}),
                NOW.isoformat(),
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
    scheduler = MemoryMaintenanceScheduler(
        repository,
        enabled=False,
        interval=timedelta(hours=18),
    )

    scheduler.tick(now=NOW + timedelta(seconds=5))
    scheduler.tick(now=NOW + timedelta(seconds=10))

    jobs = repository.queued_job_ids()
    assert len(jobs) == 1
    retry = repository.get_job(jobs[0])
    assert retry is not None
    assert json.loads(retry.payload_json)["consolidation_id"] == "con-1"
    assert json.loads(retry.payload_json)["_retry_count"] == 1
    assert repository.get_outbox_for_job(retry.job_id).state == "pending"


def test_scheduler_requeues_failed_consolidation_without_new_user_turn(tmp_path) -> None:
    repository = _repository(tmp_path)
    with connect_database(repository.database) as connection:
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 0)",
            ("feishu:chat-1", "feishu", "chat-1", NOW.isoformat(), NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO agent_jobs(job_id, kind, priority, session_key, idempotency_key, "
            "state, activity_version, payload_json, created_at, updated_at, finished_at) "
            "VALUES ('consolidate-0', 'memory.consolidate', 2, 'feishu:chat-1', "
            "'consolidate-0', 'failed', 0, '{}', ?, ?, ?)",
            (NOW.isoformat(), NOW.isoformat(), NOW.isoformat()),
        )
    scheduler = MemoryMaintenanceScheduler(
        repository,
        enabled=False,
        interval=timedelta(hours=18),
    )

    scheduler.tick(now=NOW + timedelta(seconds=5))

    jobs = repository.queued_job_ids()
    assert len(jobs) == 1
    retry = repository.get_job(jobs[0])
    assert retry is not None and retry.kind == "memory.consolidate"
    assert repository.get_outbox_for_job(retry.job_id).state == "pending"
