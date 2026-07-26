from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.extensions.skills import SkillCatalog
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.proactive.content_turn import ContentTurnResult
from memopilot.proactive.mcp_sources import ProactiveEvent, ProactiveFetchResult
from memopilot.proactive.service import ProactiveService
from memopilot.proactive.store import ProactiveRepository

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)
SESSION = "feishu:chat-1"


class _Gateway:
    def __init__(self, repository: ProactiveRepository, events: tuple[ProactiveEvent, ...]):
        self.repository = repository
        self.events = events

    async def collect(self, *, session_key: str, fetched_at: datetime):
        return self.repository.commit_fetch(
            session_key=session_key,
            source_id="feed",
            result=ProactiveFetchResult(self.events),
            fetched_at=fetched_at,
        )

    async def replay_pending_acknowledgements(self, *, now: datetime):
        return None


@dataclass
class _ContentTurn:
    result: ContentTurnResult
    candidates: tuple = ()

    async def run(self, tick_input, **kwargs):
        del kwargs
        self.candidates = tuple(tick_input.contents)
        return self.result


def _event(index: int) -> ProactiveEvent:
    return ProactiveEvent(
        "feed",
        f"e{index}",
        "content",
        NOW.isoformat(),
        {"title": f"标题 {index}", "url": f"https://example.com/{index}"},
    )


def _repository(tmp_path: Path) -> ProactiveRepository:
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    return ProactiveRepository(database)


def _service(tmp_path: Path, result: ContentTurnResult):
    repository = _repository(tmp_path)
    content_turn = _ContentTurn(result)
    service = ProactiveService(
        repository,
        _Gateway(repository, tuple(_event(index) for index in range(1, 7))),  # type: ignore[arg-type]
        content_turn,  # type: ignore[arg-type]
        skill_catalog=SkillCatalog(),
        assert_current=lambda: None,
        active_timezone=UTC,
        ordinary_cooldown=timedelta(0),
    )
    return service, repository, content_turn


def _pending_ttls(repository: ProactiveRepository) -> set[tuple[str, int]]:
    return {
        (item.source_event_id, item.ttl_hours)
        for item in repository.list_pending_acknowledgements(NOW)
    }


@pytest.mark.asyncio
async def test_content_uses_first_five_without_linear_scoring_and_preserves_ack_ttls(
    tmp_path: Path,
) -> None:
    service, repository, turn = _service(
        tmp_path,
        ContentTurnResult(
            "reply",
            message="值得看",
            cited_item_ids=("feed:e1",),
            interesting_item_ids=frozenset({"feed:e1", "feed:e2"}),
            discarded_item_ids=frozenset({"feed:e3", "feed:e4", "feed:e5"}),
        ),
    )

    outcome = await service.execute("job-1", SESSION, "chat-1", 3, NOW)

    assert [item.item_id for item in turn.candidates] == [
        "feed:e1",
        "feed:e2",
        "feed:e3",
        "feed:e4",
        "feed:e5",
    ]
    assert outcome.action == "send"
    assert outcome.source_events == (("feed", "e1"), ("feed", "e2"))
    assert [item.source_event_id for item in repository.list_unconsumed(SESSION)] == [
        "e1",
        "e2",
        "e6",
    ]
    assert _pending_ttls(repository) == {("e3", 720), ("e4", 720), ("e5", 720)}

    assert service.finalize_confirmed(outcome, confirmed_at=NOW)
    assert [item.source_event_id for item in repository.list_unconsumed(SESSION)] == ["e6"]
    assert _pending_ttls(repository) == {
        ("e1", 168),
        ("e2", 24),
        ("e3", 720),
        ("e4", 720),
        ("e5", 720),
    }


@pytest.mark.asyncio
async def test_known_delivery_failure_only_consumes_discarded_content(tmp_path: Path) -> None:
    service, repository, _ = _service(
        tmp_path,
        ContentTurnResult(
            "reply",
            message="值得看",
            cited_item_ids=("feed:e1",),
            interesting_item_ids=frozenset({"feed:e1"}),
            discarded_item_ids=frozenset({"feed:e2", "feed:e3", "feed:e4", "feed:e5"}),
        ),
    )

    outcome = await service.execute("job-1", SESSION, "chat-1", 3, NOW)
    assert service.finalize_failed(outcome)

    assert [item.source_event_id for item in repository.list_unconsumed(SESSION)] == ["e1", "e6"]
    assert _pending_ttls(repository) == {
        ("e2", 720),
        ("e3", 720),
        ("e4", 720),
        ("e5", 720),
    }


@pytest.mark.asyncio
async def test_same_source_url_is_suppressed_after_confirmed_delivery(tmp_path: Path) -> None:
    result = ContentTurnResult(
        "reply",
        message="同一事件的新文案",
        cited_item_ids=("feed:e1",),
        interesting_item_ids=frozenset({"feed:e1"}),
        discarded_item_ids=frozenset({"feed:e2", "feed:e3", "feed:e4", "feed:e5"}),
    )
    service, repository, turn = _service(tmp_path, result)
    first = await service.execute("job-1", SESSION, "chat-1", 3, NOW)
    assert service.finalize_confirmed(first, confirmed_at=NOW)

    turn.result = ContentTurnResult(
        "reply",
        message="换一种说法",
        cited_item_ids=("feed:e7",),
        interesting_item_ids=frozenset({"feed:e7"}),
        discarded_item_ids=frozenset({"feed:e6"}),
    )
    service._source_gateway.events = (
        ProactiveEvent(
            "feed",
            "e7",
            "content",
            NOW.isoformat(),
            {"title": "同一内容", "url": "https://example.com/1/"},
        ),
    )

    duplicate = await service.execute("job-2", SESSION, "chat-1", 3, NOW)

    assert duplicate.action == "quiet"
    assert duplicate.reason == "already_sent_similar"
    assert ("e7", 24) in _pending_ttls(repository)
