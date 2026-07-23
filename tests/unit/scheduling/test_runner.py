import json
from datetime import UTC, datetime, timedelta

import pytest

from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.scheduling.contracts import DueScanResult
from memopilot.scheduling.runner import SchedulerProcess, SystemScheduler
from memopilot.tasks.operational import (
    MultiplePrivateSessionsError,
    OperationalRepository,
)

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _repository(tmp_path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


def _add_session(
    repository: OperationalRepository,
    *,
    session_key: str = "feishu:chat-1",
    chat_id: str = "chat-1",
    activity_version: int = 4,
) -> None:
    with connect_database(repository.database) as connection:
        now = NOW.isoformat()
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES (?, 'feishu', ?, ?, ?)",
            (session_key, chat_id, now, now),
        )
        connection.execute(
            "INSERT INTO session_activity(session_key, activity_version, updated_at) "
            "VALUES (?, ?, ?)",
            (session_key, activity_version, now),
        )


class _MemoryScheduler:
    def __init__(self) -> None:
        self.calls: list[datetime] = []

    def tick(self, *, now: datetime):
        self.calls.append(now)
        return "memory-result"


class _ScheduleService:
    def __init__(self) -> None:
        self.calls: list[datetime] = []

    def scan_due(self, *, now: datetime) -> DueScanResult:
        self.calls.append(now)
        return DueScanResult()


def _runner(repository: OperationalRepository):
    memory = _MemoryScheduler()
    schedules = _ScheduleService()
    runner = SystemScheduler(
        repository,
        memory_scheduler=memory,
        schedule_service=schedules,
        proactive_tick_seconds=300,
    )
    return runner, memory, schedules


def test_same_proactive_bucket_enqueues_only_one_p2_job(tmp_path) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)
    runner, memory, schedules = _runner(repository)

    first = runner.tick(now=NOW)
    second = runner.tick(now=NOW + timedelta(seconds=299))

    assert first.proactive is not None and first.proactive.created is True
    assert second.proactive is not None and second.proactive.created is False
    assert memory.calls == [NOW, NOW + timedelta(seconds=299)]
    assert schedules.calls == [NOW, NOW + timedelta(seconds=299)]
    with connect_database(repository.database) as connection:
        job = connection.execute(
            "SELECT kind, priority, session_key, activity_version, payload_json "
            "FROM agent_jobs WHERE kind = 'proactive.tick'"
        ).fetchone()
        assert tuple(job[:4]) == ("proactive.tick", 2, "feishu:chat-1", 4)
        assert json.loads(job[4]) == {
            "bucket": int(NOW.timestamp() // 300),
            "chat_id": "chat-1",
        }
        assert connection.execute(
            "SELECT COUNT(*) FROM outbox_events WHERE aggregate_id = ?",
            (first.proactive.job_id,),
        ).fetchone()[0] == 1


def test_next_proactive_bucket_enqueues_new_job(tmp_path) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)
    runner, _, _ = _runner(repository)

    first = runner.tick(now=NOW)
    second = runner.tick(now=NOW + timedelta(seconds=300))

    assert first.proactive is not None and second.proactive is not None
    assert first.proactive.job_id != second.proactive.job_id
    assert repository.count("agent_jobs") == 2


def test_no_private_session_skips_proactive_but_runs_other_schedulers(tmp_path) -> None:
    repository = _repository(tmp_path)
    runner, memory, schedules = _runner(repository)

    result = runner.tick(now=NOW)

    assert result.proactive is None
    assert memory.calls == [NOW]
    assert schedules.calls == [NOW]


def test_disabled_proactive_does_not_enqueue_even_when_private_session_exists(tmp_path) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)
    memory = _MemoryScheduler()
    schedules = _ScheduleService()
    runner = SystemScheduler(
        repository,
        memory_scheduler=memory,
        schedule_service=schedules,
        proactive_tick_seconds=300,
        proactive_enabled=False,
    )

    result = runner.tick(now=NOW)

    assert result.proactive is None
    assert repository.count("agent_jobs") == 0


def test_multiple_private_sessions_are_rejected_explicitly(tmp_path) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)
    _add_session(repository, session_key="feishu:chat-2", chat_id="chat-2")
    runner, memory, schedules = _runner(repository)

    with pytest.raises(MultiplePrivateSessionsError, match="只支持一个飞书私聊"):
        runner.tick(now=NOW)

    assert memory.calls == [NOW]
    assert schedules.calls == [NOW]
    assert repository.count("agent_jobs") == 0


async def test_run_forever_uses_injected_sleep_and_clock(tmp_path) -> None:
    repository = _repository(tmp_path)
    runner, memory, _ = _runner(repository)
    sleeps: list[float] = []

    async def stop_after_first_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        raise StopAsyncIteration

    runner.sleep = stop_after_first_sleep
    runner.clock = lambda: NOW

    with pytest.raises(StopAsyncIteration):
        await runner.run_forever()

    assert memory.calls == [NOW]
    assert sleeps == [5.0]


async def test_scheduler_process_ticks_then_publishes_all_pending_outbox(tmp_path) -> None:
    repository = _repository(tmp_path)
    runner, memory, _ = _runner(repository)
    calls: list[datetime] = []

    class _Outbox:
        async def dispatch_one(self, *, now: datetime) -> bool:
            calls.append(now)
            return len(calls) < 3

    process = SchedulerProcess(runner, _Outbox(), max_publish_per_tick=10)

    published = await process.run_once(now=NOW)

    assert published == 2
    assert memory.calls == [NOW]
    assert calls == [NOW, NOW, NOW]
