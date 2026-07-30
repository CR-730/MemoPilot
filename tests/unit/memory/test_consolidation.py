from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.tasks.operational import LostLeaseError, OperationalRepository

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _Extractor:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, conversation: str) -> dict[str, object]:
        self.calls += 1
        assert "问题-1" in conversation and "回答-2" in conversation
        return {
            "history_entries": [
                {
                    "summary": "[2026-07-14 20:00] 用户完成了阶段四。",
                    "emotional_weight": 0,
                }
            ],
            "pending_items": ["- [preference] 用户偏好先读文档再实现。"],
        }


class _RecentContext:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def compress(self, **kwargs: str) -> str:
        self.calls.append(kwargs)
        return (
            "# Recent Context\n\n## Compression\n"
            "until: 2026-07-14T12:00:00+00:00\n"
            "- 最近持续关注：阶段四\n\n## Ongoing Threads\n- none\n\n"
            "## Recent Turns\n<!-- a-preview = assistant reply preview only -->\n"
            "[user] 问题-3\n"
        )


class _HistoryEntryExtractor:
    async def extract(self, conversation: str) -> dict[str, object]:
        return {
            "history_entries": [
                {
                    "summary": "[2026-07-14 20:30] 用户完成了阶段四验收。",
                    "emotional_weight": 6,
                }
            ],
            "pending_items": ["- [preference] 用户偏好先读文档再实现。"],
        }


def test_daily_journal_rejects_path_escape_and_is_idempotent(tmp_path: Path) -> None:
    markdown = MarkdownMemoryStore(tmp_path / "memory")
    content = "用户完成了阶段四。"
    content_hash = markdown.content_hash(content)

    assert markdown.append_journal(
        "2026-07-14",
        content,
        consolidation_id="con-1",
        content_hash=content_hash,
    )
    assert not markdown.append_journal(
        "2026-07-14",
        "重复内容不应写入。",
        consolidation_id="con-1",
        content_hash=content_hash,
    )
    with pytest.raises(ValueError, match="日期"):
        markdown.append_journal(
            "../bad",
            "越界",
            consolidation_id="con-2",
            content_hash=markdown.content_hash("越界"),
        )
    assert markdown.read_journal("2026-07-14").count("memopilot:con-1") == 1


def _committed_turn(
    repository: OperationalRepository,
    *,
    index: int,
    user: str,
    assistant: str,
) -> None:
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        user,
        timestamp=NOW,
        metadata={
            "event_id": f"event-{index}",
            "message_id": f"inbound-{index}",
        },
    )
    repository.record_inbound_activity(message)
    repository.commit_turn(
        message,
        assistant_content=assistant,
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
        recent_turn_count=1,
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
async def test_consolidation_writes_independent_recent_context_and_daily_journal(
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
    recent = _RecentContext()
    markdown = MarkdownMemoryStore(tmp_path / "memory")
    service = ConsolidationService(
        database,
        markdown,
        _Extractor(),
        recent_context=recent,
        keep_count=2,
        min_new_messages=4,
        recent_turn_count=1,
    )

    await service.run("feishu:chat-1")

    assert len(recent.calls) == 1
    assert markdown.read("RECENT_CONTEXT.md").startswith("# Recent Context")
    journal = markdown.read_journal("2026-07-14")
    assert journal.startswith("# 2026-07-14")
    assert "用户完成了阶段四" in journal


@pytest.mark.asyncio
async def test_below_threshold_refreshes_recent_turns_without_model_call(tmp_path: Path) -> None:
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
    markdown = MarkdownMemoryStore(tmp_path / "memory")
    markdown.replace(
        "RECENT_CONTEXT.md",
        "# Recent Context\n\n## Compression\n- 旧压缩信息\n\n## Recent Turns\n- 旧消息",
    )
    extractor = _Extractor()
    service = ConsolidationService(
        database,
        markdown,
        extractor,
        keep_count=4,
        min_new_messages=5,
        recent_turn_count=2,
    )

    result = await service.run("feishu:chat-1")

    assert result is None
    assert extractor.calls == 0
    recent = markdown.read("RECENT_CONTEXT.md")
    assert "旧压缩信息" in recent
    assert "旧消息" not in recent
    assert "[user] 问题-2" in recent
    assert "[a-preview] 回答-2" in recent
    assert "问题-1" not in recent


@pytest.mark.asyncio
async def test_history_entries_are_single_source_for_history_and_journal(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    _committed_turn(repository, index=1, user="问题-1", assistant="回答-1")
    markdown = MarkdownMemoryStore(tmp_path / "memory")
    service = ConsolidationService(
        database,
        markdown,
        _HistoryEntryExtractor(),
        keep_count=0,
        min_new_messages=1,
        recent_turn_count=1,
    )

    result = await service.run("feishu:chat-1")

    assert result is not None
    assert "[2026-07-14 20:30] 用户完成了阶段四验收。" in markdown.read("HISTORY.md")
    journal = markdown.read_journal("2026-07-14")
    assert "[2026-07-14 20:30] 用户完成了阶段四验收。" in journal
    with connect_database(database) as connection:
        output = json.loads(
            connection.execute("SELECT model_output_json FROM consolidation_manifests").fetchone()[
                0
            ]
        )
    assert output["history_entries"][0]["emotional_weight"] == 6
    assert "memories" not in output


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
        recent_turn_count=1,
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

    resumed = ConsolidationService(
        database,
        markdown,
        extractor,
        keep_count=0,
        min_new_messages=1,
        recent_turn_count=1,
    )
    result = await resumed.run("feishu:chat-1")
    repeated = await resumed.run("feishu:chat-1")

    assert result is not None and repeated is None
    assert extractor.calls == 1
    assert "用户偏好先读文档" in markdown.read("PENDING.md")
    with connect_database(database) as connection:
        manifest = connection.execute("SELECT * FROM consolidation_manifests").fetchone()
        assert manifest["state"] == "committed"


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
        recent_turn_count=1,
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
        recent_turn_count=1,
    )

    with pytest.raises(LostLeaseError, match="已失效"):
        await service.run(
            "feishu:chat-1",
            assert_current=lambda: None,
            lease=stale_lease,
        )


@pytest.mark.asyncio
async def test_29_messages_only_refreshes_explicit_recent_turn_count(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    for index in range(1, 16):
        _committed_turn(repository, index=index, user=f"问题-{index}", assistant=f"回答-{index}")
    with connect_database(database) as connection:
        connection.execute("UPDATE sessions SET last_consolidated_position = 1")
    markdown = MarkdownMemoryStore(tmp_path / "memory")
    extractor = _Extractor()
    service = ConsolidationService(
        database, markdown, extractor, keep_count=20, min_new_messages=10, recent_turn_count=10
    )

    assert await service.run("feishu:chat-1") is None
    assert extractor.calls == 0
    assert "问题-15" in markdown.read("RECENT_CONTEXT.md")
    assert "问题-10" not in markdown.read("RECENT_CONTEXT.md")


@pytest.mark.asyncio
async def test_30_messages_consolidates_old_ten_and_keeps_hot_twenty(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    for index in range(1, 16):
        _committed_turn(repository, index=index, user=f"问题-{index}", assistant=f"回答-{index}")
    service = ConsolidationService(
        database,
        MarkdownMemoryStore(tmp_path / "memory"),
        _Extractor(),
        keep_count=20,
        min_new_messages=10,
        recent_turn_count=10,
    )

    result = await service.run("feishu:chat-1")

    assert result is not None and result.message_count == 10
    with connect_database(database) as connection:
        row = connection.execute("SELECT last_consolidated_position FROM sessions").fetchone()
    assert row is not None and row[0] == 10


def test_recent_turn_count_is_required_and_positive(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    markdown = MarkdownMemoryStore(tmp_path / "memory")

    with pytest.raises(TypeError):
        ConsolidationService(database, markdown, _Extractor())
    with pytest.raises(ValueError, match="recent_turn_count"):
        ConsolidationService(database, markdown, _Extractor(), recent_turn_count=0)


def test_recent_turns_are_limited_by_keep_count_and_empty_when_keep_is_zero(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    for index in range(1, 3):
        _committed_turn(repository, index=index, user=f"问题-{index}", assistant=f"回答-{index}")

    limited = ConsolidationService(
        database,
        MarkdownMemoryStore(tmp_path / "memory"),
        _Extractor(),
        keep_count=2,
        min_new_messages=1,
        recent_turn_count=10,
    )
    empty = ConsolidationService(
        database,
        MarkdownMemoryStore(tmp_path / "empty"),
        _Extractor(),
        keep_count=0,
        min_new_messages=1,
        recent_turn_count=1,
    )

    assert "问题-2" in limited._recent_turns("feishu:chat-1")
    assert "问题-1" not in limited._recent_turns("feishu:chat-1")
    assert empty._recent_turns("feishu:chat-1") == ""


@pytest.mark.asyncio
async def test_consolidation_excludes_persisted_tool_chain_results(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "问题",
        timestamp=NOW,
        metadata={"message_id": "tool-chain-consolidation"},
    )
    repository.record_inbound_activity(message)
    repository.commit_turn(
        message,
        assistant_content="回答",
        assistant_tool_chain=(
            {
                "calls": [
                    {
                        "call_id": "call-1",
                        "name": "tool",
                        "arguments": {},
                        "result": "SECRET_TOOL_RESULT",
                    }
                ]
            },
        ),
    )

    class CapturingExtractor:
        conversation = ""

        async def extract(self, conversation: str) -> dict[str, object]:
            self.conversation = conversation
            return {}

    extractor = CapturingExtractor()
    service = ConsolidationService(
        database,
        MarkdownMemoryStore(tmp_path / "memory"),
        extractor,
        keep_count=0,
        min_new_messages=1,
        recent_turn_count=1,
    )

    assert await service.run("feishu:chat-1") is not None
    assert "问题" in extractor.conversation and "回答" in extractor.conversation
    assert "SECRET_TOOL_RESULT" not in extractor.conversation
