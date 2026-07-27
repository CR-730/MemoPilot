from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.scheduling.contracts import CreateSchedule
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.tasks.operational import LostLeaseError, OperationalRepository

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _repository(tmp_path: Path) -> ScheduleRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    with connect_database(database) as connection:
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-1', 'feishu', 'chat-1', ?, ?)",
            (NOW.isoformat(), NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO session_activity(session_key, activity_version, updated_at) "
            "VALUES ('feishu:chat-1', 3, ?)",
            (NOW.isoformat(),),
        )
    return ScheduleRepository(database)


def _create(
    repository: ScheduleRepository,
    *,
    next_run_at: datetime = NOW,
    kind: str = "after",
    expression: str = "5m",
):
    return repository.create(
        CreateSchedule(
            session_key="feishu:chat-1",
            schedule_kind=kind,  # type: ignore[arg-type]
            schedule_expression=expression,
            execution_mode="instant",
            payload={"message": "喝水"},
            next_run_at=next_run_at,
            created_at=NOW - timedelta(minutes=5),
            name="喝水",
        )
    )


def test_create_list_and_cancel_are_session_scoped(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    task = _create(repository)

    assert repository.list_for_session("feishu:chat-1") == (task,)
    assert repository.cancel("feishu:other", task_id=task.task_id, now=NOW) == ()
    assert repository.cancel("feishu:chat-1", task_id=task.task_id, now=NOW) == (
        task.task_id,
    )
    assert repository.list_for_session("feishu:chat-1") == ()


def test_due_scan_returns_stable_redis_task_and_can_recover_unpublished_task(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _create(repository)

    first = repository.enqueue_due(now=NOW)
    recovered = repository.enqueue_due(now=NOW + timedelta(seconds=1))

    assert len(first.tasks) == 1
    assert recovered.tasks[0].task_id == first.tasks[0].task_id
    assert first.tasks[0].kind == "schedule.run"
    assert first.tasks[0].payload["channel"] == "feishu"
    assert first.tasks[0].payload["activity_version"] == 3


def test_cancel_marks_queued_execution_cancelled(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    task = _create(repository)
    execution = repository.enqueue_due(now=NOW).queued[0]

    repository.cancel("feishu:chat-1", task_id=task.task_id, now=NOW)

    with connect_database(repository.database) as connection:
        state = connection.execute(
            "SELECT state FROM scheduled_executions WHERE execution_id = ?",
            (execution.execution_id,),
        ).fetchone()[0]
    assert state == "cancelled"


def test_every_coalesces_missed_windows_and_advances_to_future(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    task = _create(
        repository,
        next_run_at=NOW - timedelta(hours=3),
        kind="every",
        expression="1h",
    )

    result = repository.enqueue_due(now=NOW)

    assert result.queued[0].scheduled_at == NOW
    refreshed = repository.get(task.task_id)
    assert refreshed is not None and refreshed.next_run_at == NOW + timedelta(hours=1)


def test_expired_one_shot_is_recorded_as_skipped(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _create(repository, next_run_at=NOW - timedelta(seconds=301))

    result = repository.enqueue_due(now=NOW, misfire_grace_seconds=300)

    assert result.queued == ()
    assert result.missed[0].state == "skipped"


def test_background_execution_transition_is_fenced_and_terminal(tmp_path: Path) -> None:
    schedules = _repository(tmp_path)
    _create(schedules)
    execution = schedules.enqueue_due(now=NOW).queued[0]
    operational = OperationalRepository(schedules.database)
    epoch = operational.allocate_fence("feishu:chat-1", owner_id="runner", now=NOW)
    lease = SimpleNamespace(session_key="feishu:chat-1", owner_id="runner", epoch=epoch)

    assert (
        operational.transition_background_schedule(
            execution.execution_id,
            lease=lease,
            outcome="running",
            now=NOW,
        )
        == "running"
    )
    assert (
        operational.transition_background_schedule(
            execution.execution_id,
            lease=lease,
            outcome="succeeded",
            now=NOW,
        )
        == "succeeded"
    )
    with pytest.raises(LostLeaseError):
        operational.transition_background_schedule(
            execution.execution_id,
            lease=SimpleNamespace(
                session_key="feishu:chat-1",
                owner_id="runner",
                epoch=epoch - 1,
            ),
            outcome="running",
            now=NOW,
        )
