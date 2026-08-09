from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.persistence.conversation import (
    ConversationRepository,
    MultiplePrivateSessionsError,
)
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.runtime.outbound import DeliveryError
from memopilot.scheduling.contracts import DueScanResult
from memopilot.scheduling.repository import ScheduleRepository
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

    class Core:
        async def run_direct(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            raise RuntimeError("provider failed")

    class Outbound:
        async def dispatch(self, value):  # type: ignore[no-untyped-def]
            return True

    repository = Repository()
    service = SchedulerService(
        repository,  # type: ignore[arg-type]
        agent_core=Core(),
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
async def test_agent_schedule_uses_core_direct_session_and_sends_once() -> None:
    class Repository:
        def __init__(self) -> None:
            self.outcomes: list[str] = []

        def transition_execution(self, _execution_id, *, outcome, now):  # type: ignore[no-untyped-def]
            del now
            self.outcomes.append(outcome)
            return None

    class Core:
        def __init__(self) -> None:
            self.kwargs = None

        async def run_direct(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            react = type("React", (), {"infrastructure_error": False})()
            return type(
                "Result",
                (),
                {"reply": "done", "react": react},
            )()

    class Outbound:
        def __init__(self) -> None:
            self.calls = []

        async def dispatch(self, value):  # type: ignore[no-untyped-def]
            self.calls.append(value)
            return True

    repository, core, outbound = Repository(), Core(), Outbound()
    service = SchedulerService(repository, agent_core=core, outbound=outbound)  # type: ignore[arg-type]
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

    assert await service.execute_task(task, now=NOW) == ()
    assert core.kwargs["session_key"] == "scheduler:e"
    assert len(outbound.calls) == 1
    assert repository.outcomes == ["running", "succeeded"]


@pytest.mark.asyncio
async def test_schedule_delivery_unknown_enters_review_and_replay_does_not_repeat() -> None:
    class Repository:
        def __init__(self) -> None:
            self.state = "queued"

        def transition_execution(self, _execution_id, *, outcome, now):  # type: ignore[no-untyped-def]
            del now
            if self.state == "needs_review":
                return self.state
            self.state = outcome
            return outcome

    class Core:
        def __init__(self) -> None:
            self.calls = 0

        async def run_direct(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            react = type("React", (), {"infrastructure_error": False})()
            return type("Result", (), {"reply": "done", "react": react})()

    class Outbound:
        def __init__(self) -> None:
            self.calls = 0

        async def dispatch(self, _value):  # type: ignore[no-untyped-def]
            self.calls += 1
            return False

    repository, core, outbound = Repository(), Core(), Outbound()
    service = SchedulerService(repository, agent_core=core, outbound=outbound)  # type: ignore[arg-type]
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

    with pytest.raises(DeliveryError):
        await service.execute_task(task, now=NOW)
    assert repository.state == "needs_review"
    assert await service.execute_task(task, now=NOW + timedelta(minutes=1)) == ()
    assert core.calls == 1
    assert outbound.calls == 1


@pytest.mark.asyncio
async def test_real_schedule_repository_persists_needs_review_terminal(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    with connect_database(database) as connection:
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('cli:chat', 'cli', 'chat', ?, ?)",
            (NOW.isoformat(), NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO scheduled_tasks(task_id, session_key, schedule_kind, schedule_json, "
            "execution_mode, payload_json, next_run_at, enabled, version, created_at, updated_at) "
            "VALUES ('scheduled', 'cli:chat', 'at', '{}', 'agent', '{}', NULL, 1, 1, ?, ?)",
            (NOW.isoformat(), NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO scheduled_executions(execution_id, task_id, scheduled_at, state, "
            "created_at, updated_at) VALUES ('e', 'scheduled', ?, 'queued', ?, ?)",
            (NOW.isoformat(), NOW.isoformat(), NOW.isoformat()),
        )

    class Core:
        def __init__(self) -> None:
            self.calls = 0

        async def run_direct(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            react = type("React", (), {"infrastructure_error": False})()
            return type("Result", (), {"reply": "done", "react": react})()

    class Outbound:
        def __init__(self) -> None:
            self.calls = 0

        async def dispatch(self, _value):  # type: ignore[no-untyped-def]
            self.calls += 1
            return False

    core, outbound = Core(), Outbound()
    service = SchedulerService(
        ScheduleRepository(database), agent_core=core, outbound=outbound
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

    with pytest.raises(DeliveryError):
        await service.execute_task(task, now=NOW)
    with connect_database(database) as connection:
        state = connection.execute(
            "SELECT state FROM scheduled_executions WHERE execution_id = 'e'"
        ).fetchone()[0]
    assert state == "needs_review"
    assert await service.execute_task(task, now=NOW + timedelta(minutes=1)) == ()
    assert core.calls == 1
    assert outbound.calls == 1


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
        agent_core=object(),
        outbound=object(),
    )  # type: ignore[arg-type]
    repository = service.repository
    task = AgentTask("t", "schedule.run", 1, "cli:chat", {"execution_id": "e", **payload}, NOW)
    with pytest.raises(ValueError):
        await service.execute_task(task, now=NOW)
    assert repository.outcomes == ["running", "failed"]
