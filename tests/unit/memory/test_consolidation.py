from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.tasks.operational import InboundCommand, LostLeaseError, OperationalRepository

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _Extractor:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, conversation: str) -> dict[str, object]:
        self.calls += 1
        assert "问题-1" in conversation and "回答-2" in conversation
        return {
            "artifacts": {
                "HISTORY.md": "## 2026-07-14\n\n用户完成了阶段四。",
                "PENDING.md": "- [preference] 用户偏好先读文档再实现。",
            },
            "memories": [
                {
                    "kind": "preference",
                    "summary": "用户偏好先读文档再实现。",
                    "emotional_weight": 1,
                }
            ],
        }


def _committed_turn(
    repository: OperationalRepository,
    *,
    index: int,
    user: str,
    assistant: str,
) -> None:
    accepted = repository.accept_inbound(
        InboundCommand(
            event_id=f"event-{index}",
            message_id=f"inbound-{index}",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": user},
            received_at=NOW,
        )
    )
    owner = f"worker-{index}"
    epoch = repository.allocate_fence("feishu:chat-1", owner_id=owner, now=NOW)
    lease = SimpleNamespace(session_key="feishu:chat-1", owner_id=owner, epoch=epoch)
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    repository.commit_successful_turn(
        claim.run_id,
        lease=lease,
        user_content=user,
        assistant_content=assistant,
        now=NOW,
    )


@pytest.mark.asyncio
async def test_consolidation_window_keeps_recent_messages_and_requires_minimum(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    for index in range(1, 4):
        _committed_turn(
            repository,
            index=index,
            user=f"问题-{index}",
            assistant=f"回答-{index}",
        )
    extractor = _Extractor()
    service = ConsolidationService(
        database,
        MarkdownMemoryStore(tmp_path / "memory"),
        extractor,
        keep_count=2,
        min_new_messages=4,
    )

    result = await service.run("feishu:chat-1")

    assert result is not None
    assert result.first_position == 1
    assert result.last_position == 4
    assert result.message_count == 4
    with connect_database(database) as connection:
        last_position = connection.execute(
            "SELECT last_consolidated_position FROM sessions WHERE session_key = ?",
            ("feishu:chat-1",),
        ).fetchone()[0]
    assert last_position == 4
    assert extractor.calls == 1


@pytest.mark.asyncio
async def test_manifest_resumes_only_missing_artifact_and_publishes_vectorize_once(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    for index in range(1, 3):
        _committed_turn(
            repository,
            index=index,
            user=f"问题-{index}",
            assistant=f"回答-{index}",
        )
    extractor = _Extractor()
    markdown = MarkdownMemoryStore(tmp_path / "memory")

    def failpoint(name: str) -> None:
        if name == "after_artifact:HISTORY.md":
            raise RuntimeError("模拟文件替换后崩溃")

    service = ConsolidationService(
        database,
        markdown,
        extractor,
        keep_count=0,
        min_new_messages=1,
        failpoint=failpoint,
    )
    with pytest.raises(RuntimeError, match="崩溃"):
        await service.run("feishu:chat-1")

    assert "用户完成了阶段四" in markdown.read("HISTORY.md")
    assert markdown.read("PENDING.md") == ""
    with connect_database(database) as connection:
        manifest = connection.execute("SELECT * FROM consolidation_manifests").fetchone()
        assert manifest["state"] == "writing"
        assert json.loads(manifest["artifact_states_json"])["HISTORY.md"] == "written"
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM agent_jobs WHERE kind = 'memory.vectorize'"
            ).fetchone()[0]
            == 0
        )

    resumed = ConsolidationService(
        database,
        markdown,
        extractor,
        keep_count=0,
        min_new_messages=1,
    )
    result = await resumed.run("feishu:chat-1")
    repeated = await resumed.run("feishu:chat-1")

    assert result is not None and repeated is None
    assert extractor.calls == 1
    assert "用户偏好先读文档" in markdown.read("PENDING.md")
    with connect_database(database) as connection:
        manifest = connection.execute("SELECT * FROM consolidation_manifests").fetchone()
        assert manifest["state"] == "committed"
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM agent_jobs WHERE kind = 'memory.vectorize'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM outbox_events "
                "WHERE json_extract(payload_json, '$.kind') = 'memory.vectorize'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.asyncio
async def test_consolidation_does_not_create_manifest_after_losing_lease(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    _committed_turn(repository, index=1, user="问题-1", assistant="回答-1")
    _committed_turn(repository, index=2, user="问题-2", assistant="回答-2")
    service = ConsolidationService(
        database,
        MarkdownMemoryStore(tmp_path / "memory"),
        _Extractor(),
        keep_count=0,
        min_new_messages=1,
    )
    checks = 0

    def assert_current() -> None:
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise LostLeaseError("模拟提取期间失权")

    with pytest.raises(LostLeaseError, match="失权"):
        await service.run("feishu:chat-1", assert_current=assert_current)

    with connect_database(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM consolidation_manifests").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_consolidation_rechecks_fence_inside_final_transaction(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    _committed_turn(repository, index=1, user="问题-1", assistant="回答-1")
    _committed_turn(repository, index=2, user="问题-2", assistant="回答-2")
    stale_epoch = repository.allocate_fence("feishu:chat-1", owner_id="stale", now=NOW)
    stale_lease = SimpleNamespace(
        session_key="feishu:chat-1",
        owner_id="stale",
        epoch=stale_epoch,
    )
    repository.allocate_fence("feishu:chat-1", owner_id="current", now=NOW)
    service = ConsolidationService(
        database,
        MarkdownMemoryStore(tmp_path / "memory"),
        _Extractor(),
        keep_count=0,
        min_new_messages=1,
    )

    with pytest.raises(LostLeaseError, match="已失效"):
        await service.run(
            "feishu:chat-1",
            assert_current=lambda: None,
            lease=stale_lease,
        )

    with connect_database(database) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM agent_jobs WHERE kind = 'memory.vectorize'"
            ).fetchone()[0]
            == 0
        )
