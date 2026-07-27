from datetime import UTC, datetime, timedelta
from pathlib import Path

from memopilot.memory.scheduler import MemoryMaintenanceScheduler
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.operational import OperationalRepository

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def _repository(tmp_path: Path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


def test_same_optimizer_bucket_returns_same_redis_task_identity(tmp_path: Path) -> None:
    scheduler = MemoryMaintenanceScheduler(
        _repository(tmp_path),
        enabled=True,
        interval=timedelta(hours=18),
    )

    first = scheduler.tick(now=NOW)
    second = scheduler.tick(now=NOW + timedelta(minutes=5))

    assert first is not None and second is not None
    assert first.task_id == second.task_id
    assert first.kind == "memory.optimize"
    assert first.priority == 3


def test_disabled_optimizer_scheduler_creates_no_task(tmp_path: Path) -> None:
    scheduler = MemoryMaintenanceScheduler(
        _repository(tmp_path),
        enabled=False,
        interval=timedelta(hours=18),
    )

    assert scheduler.tick(now=NOW) is None
