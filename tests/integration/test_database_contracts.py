from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
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
from memopilot.persistence.states import EffectState, JobState, OutboxState, RunState


def test_migrate_all_databases_uses_configured_workspace(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path / "workspace", _env_file=None)

    reports = migrate_all_databases(settings)

    assert [report.to_version for report in reports] == [4, 1, 1]
    assert settings.operational_database.exists()
    assert settings.memory_database.exists()
    assert settings.wake_database.exists()


def test_operational_status_check_matches_code_enums(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path, _env_file=None)
    migrate_all_databases(settings)
    now = datetime.now(UTC).isoformat()

    with connect_database(settings.operational_database) as connection:
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("feishu:chat", "feishu", "chat", now, now),
        )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            connection.execute(
                """
                INSERT INTO agent_jobs(
                    job_id, kind, priority, session_key, idempotency_key, state,
                    activity_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("job", "turn", 0, "feishu:chat", "inbound:event", "invalid", 1, now, now),
            )

    assert {state.value for state in JobState} == {
        "queued",
        "running",
        "succeeded",
        "skipped",
        "failed",
        "cancelled",
        "needs_review",
    }
    assert "recovering" in {state.value for state in RunState}
    assert {state.value for state in OutboxState} == {
        "pending",
        "publishing",
        "published",
        "dead",
    }
    assert "unknown" in {state.value for state in EffectState}


def test_memory_source_ref_and_wake_source_event_are_idempotency_keys(tmp_path: Path) -> None:
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

    with connect_database(settings.wake_database) as connection:
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
