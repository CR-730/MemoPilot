from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path

import pytest

from memopilot.persistence.migrations import (
    DatabaseKind,
    Migration,
    MigrationError,
    UnsupportedDatabaseVersionError,
    connect_database,
    execute_with_busy_retry,
    migrate_database,
)

EXPECTED_TABLES = {
    DatabaseKind.OPERATIONAL: {
        "inbound_events",
        "sessions",
        "session_activity",
        "agent_jobs",
        "runs",
        "run_attempts",
        "steps",
        "session_fences",
        "outbox_events",
        "outbound_effects",
        "scheduled_tasks",
        "scheduled_executions",
        "consolidation_manifests",
        "messages",
        "session_identities",
        "session_interrupts",
    },
    DatabaseKind.MEMORY: {
        "memory_items",
        "memory_embeddings",
        "memory_metadata",
        "memory_relations",
        "keyword_index_metadata",
    },
    DatabaseKind.WAKE: {
        "source_events",
        "source_cursors",
        "pending_acknowledgements",
        "content_scores",
        "hazard_snapshots",
        "wake_decisions",
        "drift_history",
    },
}


@pytest.mark.parametrize("kind", list(DatabaseKind))
def test_migrations_create_expected_schema(tmp_path: Path, kind: DatabaseKind) -> None:
    database = tmp_path / f"{kind.value}.db"

    report = migrate_database(database, kind)

    assert report.from_version == 0
    expected_version = 3 if kind is DatabaseKind.OPERATIONAL else 1
    assert report.to_version == expected_version
    assert report.backup_path is None
    with connect_database(database) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        table_names = {str(row[0]) for row in table_rows}
        assert version == expected_version
        assert EXPECTED_TABLES[kind] <= table_names
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_operational_v1_upgrades_to_v2_without_losing_existing_rows(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    v1_sql = (
        files("memopilot.persistence.schema")
        .joinpath("operational_v1.sql")
        .read_text("utf-8")
    )
    with sqlite3.connect(database) as connection:
        connection.executescript(v1_sql)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-1', 'feishu', 'chat-1', 'now', 'now')"
        )

    report = migrate_database(database, DatabaseKind.OPERATIONAL)

    assert report.from_version == 1
    assert report.to_version == 3
    assert report.applied_versions == (2, 3)
    assert report.backup_path is not None
    with connect_database(database) as connection:
        session = connection.execute(
            "SELECT chat_id FROM sessions WHERE session_key = 'feishu:chat-1'"
        ).fetchone()
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(outbound_effects)").fetchall()
        }
    assert session is not None and session[0] == "chat-1"
    assert {
        "channel",
        "chat_id",
        "payload_json",
        "last_attempt_at",
        "cancel_on_activity",
    } <= columns


def test_migration_backs_up_existing_database_before_upgrade(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE legacy_marker(value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_marker VALUES ('before-migration')")

    report = migrate_database(database, DatabaseKind.OPERATIONAL)

    assert report.backup_path is not None
    assert report.backup_path.exists()
    assert report.backup_path.name.startswith("operational.db.bak-v0-")
    with sqlite3.connect(report.backup_path) as backup:
        value = backup.execute("SELECT value FROM legacy_marker").fetchone()[0]
    assert value == "before-migration"


def test_database_newer_than_supported_schema_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "wake.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(UnsupportedDatabaseVersionError, match="99"):
        migrate_database(database, DatabaseKind.WAKE)


def test_failed_migration_rolls_back_schema_and_version(tmp_path: Path) -> None:
    database = tmp_path / "broken.db"
    migrations = (
        Migration(
            version=1,
            name="broken",
            sql="CREATE TABLE should_rollback(id TEXT); INSERT INTO missing_table VALUES (1);",
        ),
    )

    with pytest.raises(MigrationError, match="broken"):
        migrate_database(database, DatabaseKind.OPERATIONAL, migrations=migrations)

    with sqlite3.connect(database) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='should_rollback'"
        ).fetchone()
    assert version == 0
    assert table is None


def test_busy_retry_is_bounded_and_only_retries_lock_errors() -> None:
    attempts = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")
        return "done"

    result = execute_with_busy_retry(
        operation,
        max_attempts=3,
        base_delay_seconds=0.01,
        sleep=delays.append,
    )

    assert result == "done"
    assert attempts == 3
    assert delays == [0.01, 0.02]

    with pytest.raises(sqlite3.OperationalError, match="syntax error"):
        execute_with_busy_retry(
            lambda: (_ for _ in ()).throw(sqlite3.OperationalError("syntax error")),
            max_attempts=3,
            sleep=delays.append,
        )
