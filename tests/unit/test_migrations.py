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
        "turn_interrupt_snapshots",
    },
    DatabaseKind.MEMORY: {
        "memory_items",
        "memory_embeddings",
        "memory_metadata",
        "memory_relations",
        "keyword_index_metadata",
        "memory_sources",
        "memory_vector_rows",
        "memory_ingestion_batches",
        "memory_fts",
        "memory_usages",
    },
    DatabaseKind.PROACTIVE: {
        "source_events",
        "source_event_history",
        "pending_acknowledgements",
        "hazard_snapshots",
        "proactive_decisions",
        "drift_history",
        "hazard_state",
        "context_states",
        "context_reevaluation_state",
        "drift_state",
        "proactive_observations",
    },
}


@pytest.mark.parametrize("kind", list(DatabaseKind))
def test_migrations_create_expected_schema(tmp_path: Path, kind: DatabaseKind) -> None:
    database = tmp_path / f"{kind.value}.db"

    report = migrate_database(database, kind)

    assert report.from_version == 0
    expected_version = {
        DatabaseKind.OPERATIONAL: 5,
        DatabaseKind.MEMORY: 3,
        DatabaseKind.PROACTIVE: 7,
    }[kind]
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


def test_operational_v1_upgrades_to_v5_without_losing_existing_rows(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    v1_sql = files("memopilot.persistence.schema").joinpath("operational_v1.sql").read_text("utf-8")
    with sqlite3.connect(database) as connection:
        connection.executescript(v1_sql)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-1', 'feishu', 'chat-1', 'now', 'now')"
        )

    report = migrate_database(database, DatabaseKind.OPERATIONAL)

    assert report.from_version == 1
    assert report.to_version == 5
    assert report.applied_versions == (2, 3, 4, 5)
    assert report.backup_path is not None
    with connect_database(database) as connection:
        session = connection.execute(
            "SELECT chat_id FROM sessions WHERE session_key = 'feishu:chat-1'"
        ).fetchone()
        effect_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(outbound_effects)").fetchall()
        }
        session_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }
        message_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
    assert session is not None and session[0] == "chat-1"
    assert {
        "channel",
        "chat_id",
        "payload_json",
        "last_attempt_at",
        "cancel_on_activity",
    } <= effect_columns
    assert "last_consolidated_position" in session_columns
    assert "session_position" in message_columns


def test_proactive_v1_upgrades_to_v7_without_losing_reservoir_rows(tmp_path: Path) -> None:
    database = tmp_path / "proactive.db"
    v1_sql = files("memopilot.persistence.schema").joinpath("proactive_v1.sql").read_text("utf-8")
    with sqlite3.connect(database) as connection:
        connection.executescript(v1_sql)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            INSERT INTO source_events(
                reservoir_id, source_id, source_event_id, session_key, kind,
                occurred_at, payload_json, fetched_at
            ) VALUES ('r1', 'news', 'e1', 'feishu:c1', 'content', 'now', '{}', 'now')
            """
        )

    report = migrate_database(database, DatabaseKind.PROACTIVE)

    assert report.from_version == 1
    assert report.to_version == 7
    assert report.applied_versions == (2, 3, 4, 5, 6, 7)
    with connect_database(database) as connection:
        event = connection.execute(
            "SELECT source_event_id FROM source_events WHERE reservoir_id = 'r1'"
        ).fetchone()
        state_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'hazard_state'"
        ).fetchone()
        decision_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(proactive_decisions)").fetchall()
        }
        ack_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(pending_acknowledgements)").fetchall()
        }
        content_scores = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'content_scores'"
        ).fetchone()
        history_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'source_event_history'"
        ).fetchone()
    assert event is not None and event[0] == "e1"
    assert state_table is not None
    assert "decision_payload_json" in decision_columns
    assert "ttl_hours" in ack_columns
    assert content_scores is None
    assert history_table is not None


def test_operational_v4_upgrades_to_v5_without_losing_existing_rows(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    with sqlite3.connect(database) as connection:
        for version in range(1, 5):
            sql = (
                files("memopilot.persistence.schema")
                .joinpath(f"operational_v{version}.sql")
                .read_text("utf-8")
            )
            connection.executescript(sql)
        connection.execute("PRAGMA user_version = 4")
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-v4', 'feishu', 'chat-v4', 'now', 'now')"
        )

    report = migrate_database(database, DatabaseKind.OPERATIONAL)

    assert report.from_version == 4
    assert report.to_version == 5
    assert report.applied_versions == (5,)
    with connect_database(database) as connection:
        session = connection.execute(
            "SELECT chat_id, last_consolidated_position FROM sessions "
            "WHERE session_key = 'feishu:chat-v4'"
        ).fetchone()
    assert session is not None
    assert tuple(session) == ("chat-v4", 0)


def test_operational_v5_backfills_existing_message_positions(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    with sqlite3.connect(database) as connection:
        for version in range(1, 5):
            sql = (
                files("memopilot.persistence.schema")
                .joinpath(f"operational_v{version}.sql")
                .read_text("utf-8")
            )
            connection.executescript(sql)
        connection.execute("PRAGMA user_version = 4")
    with connect_database(database) as connection:
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('s1', 'feishu', 'c1', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
        )
        connection.executemany(
            "INSERT INTO messages(message_id, session_key, role, content, turn_id, "
            "turn_position, created_at) VALUES (?, 's1', ?, ?, ?, ?, ?)",
            [
                ("m2", "assistant", "答复", "t1", 1, "2026-01-01T00:00:01Z"),
                ("m1", "user", "问题", "t1", 0, "2026-01-01T00:00:00Z"),
            ],
        )
        connection.commit()

    migrate_database(database, DatabaseKind.OPERATIONAL)

    with connect_database(database) as connection:
        rows = connection.execute(
            "SELECT message_id, session_position FROM messages "
            "WHERE session_key = 's1' ORDER BY session_position"
        ).fetchall()
    assert [tuple(row) for row in rows] == [("m1", 1), ("m2", 2)]


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
    database = tmp_path / "proactive.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(UnsupportedDatabaseVersionError, match="99"):
        migrate_database(database, DatabaseKind.PROACTIVE)


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
