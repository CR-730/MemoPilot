from datetime import UTC, datetime, timedelta
from pathlib import Path

from memopilot.persistence.conversation import ConversationRepository
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.scheduling.contracts import DueScanResult
from memopilot.tasks.producer import TaskProducer

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _Schedules:
    def scan_due(self, *, now: datetime) -> DueScanResult:
        del now
        return DueScanResult()


def _producer(tmp_path: Path, *, enabled: bool) -> TaskProducer:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return TaskProducer(
        ConversationRepository(database),
        schedule_service=_Schedules(),
        memory_optimizer_enabled=enabled,
        memory_optimizer_interval=timedelta(hours=18),
        proactive_tick_seconds=1800,
        proactive_enabled=False,
    )


def test_same_optimizer_bucket_returns_same_task_identity(tmp_path: Path) -> None:
    first = _producer(tmp_path, enabled=True).tick(now=NOW).memory
    second = _producer(tmp_path, enabled=True).tick(now=NOW + timedelta(minutes=5)).memory

    assert first is not None and second is not None
    assert first.task_id == second.task_id
    assert first.kind == "memory.optimize"
    assert first.priority == 3


def test_disabled_optimizer_creates_no_task(tmp_path: Path) -> None:
    assert _producer(tmp_path, enabled=False).tick(now=NOW).memory is None
