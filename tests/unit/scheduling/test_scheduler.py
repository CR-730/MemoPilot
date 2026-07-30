from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.persistence.conversation import (
    ConversationRepository,
    MultiplePrivateSessionsError,
)
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.scheduling.contracts import DueScanResult
from memopilot.scheduling.scheduler import ScheduledTurnPipeline
from memopilot.scheduling.service import SchedulerService
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.producer import TaskProducer

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _repository(tmp_path: Path) -> ConversationRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return ConversationRepository(database)


def _add_session(
    repository: ConversationRepository,
    *,
    session_key: str = "feishu:chat-1",
    chat_id: str = "chat-1",
) -> None:
    with connect_database(repository.database) as connection:
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES (?, 'feishu', ?, ?, ?)",
            (session_key, chat_id, NOW.isoformat(), NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO session_activity(session_key, activity_version, updated_at) "
            "VALUES (?, 4, ?)",
            (session_key, NOW.isoformat()),
        )


class _Schedules:
    def scan_due(self, *, now: datetime) -> DueScanResult:
        del now
        return DueScanResult()


def _scheduler(repository: ConversationRepository) -> TaskProducer:
    return TaskProducer(
        repository,
        schedule_service=_Schedules(),
        memory_optimizer_enabled=True,
        memory_optimizer_interval=timedelta(hours=1),
        proactive_tick_seconds=1800,
    )


def test_proactive_tick_is_stable_inside_same_time_bucket(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)
    scheduler = _scheduler(repository)

    first = scheduler.tick(now=NOW).proactive
    second = scheduler.tick(now=NOW + timedelta(seconds=1799)).proactive

    assert first is not None and second is not None
    assert first.task_id == second.task_id
    assert first.priority == 2
    assert first.payload["channel"] == "feishu"
    assert first.payload["activity_version"] == 4


def test_no_private_session_skips_only_proactive_task(tmp_path: Path) -> None:
    result = _scheduler(_repository(tmp_path)).tick(now=NOW)

    assert result.proactive is None
    assert isinstance(result.memory, AgentTask)


def test_multiple_private_sessions_are_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)
    _add_session(repository, session_key="feishu:chat-2", chat_id="chat-2")

    with pytest.raises(MultiplePrivateSessionsError):
        _scheduler(repository).tick(now=NOW)


async def test_scheduler_process_publishes_each_background_task_once(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)

    class Queue:
        def __init__(self) -> None:
            self.ids: list[str] = []

        async def publish_task_once(self, task: AgentTask) -> str | None:
            self.ids.append(task.task_id)
            return task.task_id

    queue = Queue()
    process = ScheduledTurnPipeline(_scheduler(repository), queue)

    assert await process.run_once(now=NOW) == 2
    assert queue.ids[0].startswith("memory.optimize:")
    assert queue.ids[1].startswith("proactive.tick:")


async def test_scheduler_publishes_proactive_without_duplicate_busy_gate(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _add_session(repository)

    class Queue:
        def __init__(self) -> None:
            self.ids: list[str] = []

        async def publish_task_once(self, task: AgentTask) -> str | None:
            self.ids.append(task.task_id)
            return task.task_id

    queue = Queue()
    process = ScheduledTurnPipeline(_scheduler(repository), queue)

    assert await process.run_once(now=NOW) == 2
    assert queue.ids[0].startswith("memory.optimize:")
    assert queue.ids[1].startswith("proactive.tick:")


@pytest.mark.asyncio
async def test_schedule_runtime_failure_marks_execution_failed_and_reraises(tmp_path: Path) -> None:
    class Repository:
        def __init__(self) -> None:
            self.outcomes = []

        def transition_execution(self, execution_id, *, outcome, now):  # type: ignore[no-untyped-def]
            self.outcomes.append(outcome)
            return None

    class Runtime:
        async def run(self, turn):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider failed")

    class Outbound:
        async def dispatch(self, value):  # type: ignore[no-untyped-def]
            return True

    repository = Repository()
    service = SchedulerService(
        repository,  # type: ignore[arg-type]
        runtime=Runtime(),
        outbound=Outbound(),
    )  # type: ignore[arg-type]
    task = AgentTask(
        "t",
        "schedule.run",
        1,
        "cli:chat",
        {
            "execution_id": "e",
            "execution_mode": "agent",
            "payload": {"prompt": "go"},
            "channel": "cli",
            "chat_id": "chat",
        },
        NOW,
    )
    with pytest.raises(RuntimeError, match="provider failed"):
        await service.execute_task(task, now=NOW)
    assert repository.outcomes == ["running", "failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"execution_mode": "agent"}, {"payload": {}}])
async def test_schedule_invalid_running_payload_marks_failed_and_reraises(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    class Repository:
        def __init__(self) -> None:
            self.outcomes: list[str] = []

        def transition_execution(self, execution_id, *, outcome, now):  # type: ignore[no-untyped-def]
            self.outcomes.append(outcome)

    service = SchedulerService(
        Repository(),  # type: ignore[arg-type]
        runtime=object(),
        outbound=object(),
    )  # type: ignore[arg-type]
    repository = service.repository
    task = AgentTask("t", "schedule.run", 1, "cli:chat", {"execution_id": "e", **payload}, NOW)
    with pytest.raises(ValueError):
        await service.execute_task(task, now=NOW)
    assert repository.outcomes == ["running", "failed"]
