from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from memopilot.bus.events import InboundMessage
from memopilot.channels.base import AttachmentStore, MessageDeduper, SessionIdentityIndex
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.operational import OperationalRepository

NOW = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)


def _repository(tmp_path: Path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


def test_attachment_store_writes_under_workspace_uploads(tmp_path: Path) -> None:
    store = AttachmentStore(tmp_path / "uploads")

    path = store.write_bytes(b"image", prefix="feishu_", suffix=".png")

    assert path.parent == tmp_path / "uploads"
    assert path.suffix == ".png"
    assert path.read_bytes() == b"image"


def test_message_deduper_evicts_oldest_key() -> None:
    deduper = MessageDeduper(max_size=2)

    assert deduper.seen("a") is False
    assert deduper.seen("b") is False
    assert deduper.seen("a") is True
    assert deduper.seen("c") is False
    assert deduper.seen("a") is False


def test_identity_index_rebuilds_and_persists_all_feishu_ids(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.record_inbound_activity(
        InboundMessage(
            "feishu",
            "user",
            "chat-1",
            "你好",
            timestamp=NOW,
            metadata={"event_id": "event-1", "message_id": "message-1"},
        )
    )
    repository.remember_session_identities(
        session_key="feishu:chat-1",
        channel="feishu",
        chat_id="chat-1",
        identities={"open_id": "ou_old"},
        now=NOW,
    )
    index = SessionIdentityIndex(repository, channel="feishu", clock=lambda: NOW)

    assert index.rebuild() == {"ou_old": "chat-1"}

    index.remember(
        session_key="feishu:chat-1",
        chat_id="chat-1",
        identities={"open_id": "ou_new", "user_id": "u_new", "union_id": "on_new"},
    )

    assert index.resolve("u_new") == "chat-1"
    rebuilt = SessionIdentityIndex(repository, channel="feishu", clock=lambda: NOW)
    assert rebuilt.rebuild().items() >= {
        "ou_new": "chat-1",
        "u_new": "chat-1",
        "on_new": "chat-1",
    }.items()
