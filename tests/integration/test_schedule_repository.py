import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from memopilot.delivery.effects import EffectRepository, EffectRequest
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.scheduling.contracts import CreateSchedule
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import LostLeaseError, OperationalRepository

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _repository(tmp_path) -> ScheduleRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    with connect_database(database) as connection:
        now = NOW.isoformat()
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-1', 'feishu', 'chat-1', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO session_activity(session_key, activity_version, last_user_at, updated_at) "
            "VALUES ('feishu:chat-1', 3, ?, ?)",
            (now, now),
        )
    return ScheduleRepository(database)


def _create(
    repository: ScheduleRepository,
    *,
    kind: str = "after",
    mode: str = "instant",
    next_run_at: datetime = NOW,
    name: str | None = "喝水",
):
    return repository.create(
        CreateSchedule(
            session_key="feishu:chat-1",
            schedule_kind=kind,
            schedule_expression="5m",
            execution_mode=mode,
            payload={"message": "喝水"} if mode == "instant" else {"prompt": "查询天气"},
            next_run_at=next_run_at,
            timezone="Asia/Shanghai",
            name=name,
            created_at=NOW - timedelta(minutes=5),
        )
    )


def test_create_list_and_cancel_are_scoped_to_session(tmp_path) -> None:
    repository = _repository(tmp_path)
    created = _create(repository)

    assert repository.list_for_session("feishu:chat-1") == (created,)
    assert repository.list_for_session("feishu:other") == ()
    assert repository.cancel("feishu:other", task_id=created.task_id) == ()
    assert repository.cancel("feishu:chat-1", task_id=created.task_id) == (created.task_id,)
    assert repository.list_for_session("feishu:chat-1") == ()


def test_create_replay_with_same_stable_parameters_returns_existing_task(tmp_path) -> None:
    repository = _repository(tmp_path)

    first = _create(repository)
    replayed = _create(repository)

    assert replayed == first
    with connect_database(repository.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM scheduled_tasks").fetchone()[0] == 1


def test_task_id_collision_with_different_parameters_is_rejected(
    tmp_path, monkeypatch
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(
        "memopilot.scheduling.repository._stable_id",
        lambda _kind, _identity: "schedule-forced-collision",
    )
    _create(repository, name="任务 A")

    with pytest.raises(ValueError, match="不同参数"):
        _create(repository, name="任务 B")


def test_due_scan_atomically_creates_unique_execution_job_and_outbox(tmp_path) -> None:
    repository = _repository(tmp_path)
    task = _create(repository)

    first = repository.enqueue_due(now=NOW)
    second = repository.enqueue_due(now=NOW)

    assert len(first.queued) == 1
    assert second.queued == ()
    execution = first.queued[0]
    assert execution.task_id == task.task_id
    with connect_database(repository.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM scheduled_executions").fetchone()[0] == 1
        job = connection.execute("SELECT kind, priority FROM agent_jobs").fetchone()
        assert tuple(job) == ("schedule.run", 1)
        assert connection.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 1
        row = connection.execute(
            "SELECT enabled, next_run_at FROM scheduled_tasks WHERE task_id = ?", (task.task_id,)
        ).fetchone()
        assert tuple(row) == (0, None)


def test_every_coalesces_missed_windows_and_advances_to_future(tmp_path) -> None:
    repository = _repository(tmp_path)
    task = repository.create(
        CreateSchedule(
            session_key="feishu:chat-1",
            schedule_kind="every",
            schedule_expression="1h",
            execution_mode="instant",
            payload={"message": "整点提醒"},
            next_run_at=NOW - timedelta(hours=3),
            timezone="UTC",
            created_at=NOW - timedelta(days=1),
        )
    )

    result = repository.enqueue_due(now=NOW)

    assert len(result.queued) == 1
    assert result.queued[0].scheduled_at == NOW
    with connect_database(repository.database) as connection:
        payload = json.loads(
            connection.execute("SELECT payload_json FROM agent_jobs").fetchone()[0]
        )
    assert payload["scheduled_at"] == NOW.isoformat()
    updated = repository.get(task.task_id)
    assert updated is not None
    assert updated.next_run_at == NOW + timedelta(hours=1)
    assert repository.enqueue_due(now=NOW).queued == ()


def test_expired_one_shot_is_recorded_as_skipped_without_job(tmp_path) -> None:
    repository = _repository(tmp_path)
    task = _create(repository, next_run_at=NOW - timedelta(seconds=301))

    result = repository.enqueue_due(now=NOW, misfire_grace_seconds=300)

    assert result.queued == ()
    assert len(result.missed) == 1
    assert result.missed[0].task_id == task.task_id
    with connect_database(repository.database) as connection:
        execution = connection.execute(
            "SELECT state FROM scheduled_executions"
        ).fetchone()
        assert execution[0] == "skipped"
        assert connection.execute("SELECT COUNT(*) FROM agent_jobs").fetchone()[0] == 0


def test_cancelled_task_is_not_scanned(tmp_path) -> None:
    repository = _repository(tmp_path)
    task = _create(repository)
    repository.cancel("feishu:chat-1", task_id=task.task_id)

    assert repository.enqueue_due(now=NOW).queued == ()


def test_cancel_after_due_scan_cancels_queued_execution_and_job(tmp_path) -> None:
    repository = _repository(tmp_path)
    task = _create(repository)
    execution = repository.enqueue_due(now=NOW).queued[0]

    cancelled = repository.cancel(
        "feishu:chat-1",
        task_id=task.task_id,
        now=NOW + timedelta(seconds=1),
    )

    assert cancelled == (task.task_id,)
    with connect_database(repository.database) as connection:
        execution_state = connection.execute(
            "SELECT state FROM scheduled_executions WHERE execution_id = ?",
            (execution.execution_id,),
        ).fetchone()[0]
        job_state = connection.execute(
            "SELECT state FROM agent_jobs WHERE job_id = ?",
            (execution.job_id,),
        ).fetchone()[0]
    assert execution_state == "cancelled"
    assert job_state == "cancelled"


def test_due_scan_rolls_back_execution_job_outbox_and_task_advance_together(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    task = _create(repository)

    def crash(point: str) -> None:
        if point == "after_job_outbox":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        repository.enqueue_due(now=NOW, failpoint=crash)

    with connect_database(repository.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM scheduled_executions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM agent_jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 0
    unchanged = repository.get(task.task_id)
    assert unchanged is not None
    assert unchanged.enabled is True
    assert unchanged.next_run_at == NOW


def test_execution_state_transition_is_fenced_and_terminal_is_idempotent(tmp_path) -> None:
    schedules = _repository(tmp_path)
    _create(schedules)
    execution = schedules.enqueue_due(now=NOW).queued[0]
    operational = OperationalRepository(schedules.database)
    lease = SimpleNamespace(
        session_key="feishu:chat-1",
        owner_id="worker-1",
        epoch=1,
    )
    with connect_database(schedules.database) as connection:
        connection.execute(
            "INSERT INTO session_fences("
            "session_key, current_epoch, owner_id, heartbeat_at, updated_at) "
            "VALUES ('feishu:chat-1', 1, 'worker-1', ?, ?)",
            (NOW.isoformat(), NOW.isoformat()),
        )

    assert operational.transition_scheduled_execution(
        execution.execution_id,
        job_id=execution.job_id or "",
        lease=lease,
        outcome="running",
        now=NOW,
    ) == "running"
    assert operational.transition_scheduled_execution(
        execution.execution_id,
        job_id=execution.job_id or "",
        lease=lease,
        outcome="succeeded",
        now=NOW,
    ) == "succeeded"
    assert operational.transition_scheduled_execution(
        execution.execution_id,
        job_id=execution.job_id or "",
        lease=lease,
        outcome="running",
        now=NOW,
    ) == "succeeded"

    stale = SimpleNamespace(
        session_key="feishu:chat-1",
        owner_id="worker-1",
        epoch=0,
    )
    with pytest.raises(LostLeaseError):
        operational.transition_scheduled_execution(
            execution.execution_id,
            job_id=execution.job_id or "",
            lease=stale,
            outcome="running",
            now=NOW,
        )


def test_manual_effect_review_also_closes_scheduled_execution(tmp_path) -> None:
    schedules = _repository(tmp_path)
    _create(schedules)
    execution = schedules.enqueue_due(now=NOW).queued[0]
    operational = OperationalRepository(schedules.database)
    epoch = operational.allocate_fence(
        "feishu:chat-1", owner_id="worker-1", now=NOW
    )
    lease = SessionLease(
        session_key="feishu:chat-1",
        owner_id="worker-1",
        epoch=epoch,
        redis_key="lease",
        redis_value=f"worker-1|epoch|{epoch}",
    )
    claim = operational.claim_job(execution.job_id or "", lease=lease, now=NOW)
    assert claim is not None
    job = operational.get_job(claim.job_id)
    assert job is not None
    operational.transition_scheduled_execution(
        execution.execution_id,
        job_id=claim.job_id,
        lease=lease,
        outcome="running",
        now=NOW,
    )
    effect_repo = EffectRepository(schedules.database)
    effect = effect_repo.create(
        EffectRequest(
            operation_id="scheduled-effect",
            run_id=claim.run_id,
            session_key=claim.session_key,
            channel="feishu",
            chat_id="chat-1",
            text="提醒",
            expected_activity_version=job.activity_version,
            lease=lease,
            now=NOW,
        )
    )
    effect_repo.begin_send(effect.operation_id, lease=lease, now=NOW)
    effect_repo.mark_unknown(
        effect.operation_id,
        lease=lease,
        error="响应丢失",
        now=NOW,
    )
    operational.transition_scheduled_execution(
        execution.execution_id,
        job_id=claim.job_id,
        lease=lease,
        outcome="needs_review",
        now=NOW,
    )
    operational.finish_job(claim.run_id, lease=lease, outcome="needs_review", now=NOW)

    operational.resolve_effect_review(
        effect.operation_id,
        lease=lease,
        decision="confirmed",
        message_id="om-confirmed",
        now=NOW + timedelta(seconds=1),
    )

    with connect_database(schedules.database) as connection:
        state = connection.execute(
            "SELECT state FROM scheduled_executions WHERE execution_id = ?",
            (execution.execution_id,),
        ).fetchone()[0]
    assert state == "succeeded"
