from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import pytest

from memopilot.config import MemoPilotSettings
from memopilot.persistence.memory_metadata import (
    EmbeddingConfigurationMismatchError,
    EmbeddingIdentity,
    ensure_embedding_identity,
)
from memopilot.persistence.migrations import (
    connect_database,
    migrate_all_databases,
)


def test_migrate_all_databases_uses_configured_workspace(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path / "workspace", _env_file=None)

    reports = migrate_all_databases(settings)

    assert [report.to_version for report in reports] == [9, 3, 9]
    assert settings.operational_database.exists()
    assert settings.memory_database.exists()
    assert settings.proactive_database.exists()


def test_migrate_all_databases_adopts_legacy_wake_v1_database(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path / "workspace", _env_file=None)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    legacy = settings.data_dir / "wake.db"
    v1_sql = (
        files("memopilot.persistence.schema")
        .joinpath("proactive_v1.sql")
        .read_text("utf-8")
        .replace("proactive_decisions", "wake_decisions")
        .replace("ix_proactive_decisions_session", "ix_wake_decisions_session")
    )
    connection = sqlite3.connect(legacy)
    try:
        connection.executescript(v1_sql)
        connection.execute(
            "CREATE TABLE source_cursors("
            "source_id TEXT PRIMARY KEY, cursor_value TEXT, updated_at TEXT NOT NULL)"
        )
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            INSERT INTO source_events(
                reservoir_id, source_id, source_event_id, session_key, kind,
                occurred_at, payload_json, fetched_at
            ) VALUES ('legacy-r1', 'feed', 'legacy-e1', 'feishu:c1',
                      'content', '2026-07-01T00:00:00+00:00', '{}',
                      '2026-07-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO wake_decisions(
                decision_id, session_key, trigger_kind, action, source_events_json,
                activity_version, state, decided_at
            ) VALUES ('legacy-d1', 'feishu:c1', 'content', 'skip',
                      '[["feed", "legacy-e1"]]', 1, 'committed',
                      '2026-07-01T00:00:00+00:00')
            """
        )
        connection.commit()
    finally:
        connection.close()

    reports = migrate_all_databases(settings)

    assert reports[-1].to_version == 9
    assert settings.proactive_database.exists()
    assert not legacy.exists()
    with connect_database(settings.proactive_database) as connection:
        assert (
            connection.execute(
                "SELECT source_event_id FROM source_events WHERE reservoir_id = 'legacy-r1'"
            ).fetchone()[0]
            == "legacy-e1"
        )
        assert (
            connection.execute(
                "SELECT decision_id FROM proactive_decisions WHERE decision_id = 'legacy-d1'"
            ).fetchone()[0]
            == "legacy-d1"
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'source_cursors'"
            ).fetchone()
            is None
        )


def test_operational_schema_has_no_job_run_or_outbox_tables(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path, _env_file=None)
    migrate_all_databases(settings)
    now = datetime.now(UTC).isoformat()

    with connect_database(settings.operational_database) as connection:
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("feishu:chat", "feishu", "chat", now, now),
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {
        "agent_jobs",
        "runs",
        "run_attempts",
        "steps",
        "outbox_events",
        "outbound_effects",
    }.isdisjoint(tables)

def test_memory_source_ref_and_proactive_source_event_are_idempotency_keys(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path, _env_file=None)
    migrate_all_databases(settings)
    now = datetime.now(UTC).isoformat()

    with connect_database(settings.memory_database) as connection:
        values = ("event-1", "event", "summary", "hash-1", "consolidation:1", now, now)
        connection.execute(
            """
            INSERT INTO memory_items(
                item_id, memory_type, summary, content_hash, source_ref, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            connection.execute(
                """
                INSERT INTO memory_items(
                    item_id, memory_type, summary, content_hash, source_ref,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("event-2", "event", "other", "hash-2", "consolidation:1", now, now),
            )

    with connect_database(settings.proactive_database) as connection:
        event = ("row-1", "feed", "article-1", "feishu:chat", "content", now, "{}", now)
        connection.execute(
            """
            INSERT INTO source_events(
                reservoir_id, source_id, source_event_id, session_key, kind,
                occurred_at, payload_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            event,
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            connection.execute(
                """
                INSERT INTO source_events(
                    reservoir_id, source_id, source_event_id, session_key, kind,
                    occurred_at, payload_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("row-2", *event[1:]),
            )


def test_embedding_identity_is_written_once_and_mismatch_fails_fast(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path, _env_file=None)
    migrate_all_databases(settings)
    identity = EmbeddingIdentity(
        base_url="https://embedding.example/v1",
        model="embedding-model",
        dimension=1024,
    )

    ensure_embedding_identity(settings.memory_database, identity)
    ensure_embedding_identity(settings.memory_database, identity)

    with pytest.raises(EmbeddingConfigurationMismatchError, match="不匹配"):
        ensure_embedding_identity(
            settings.memory_database,
            EmbeddingIdentity(
                base_url=identity.base_url,
                model=identity.model,
                dimension=1536,
            ),
        )
