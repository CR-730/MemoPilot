from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.extensions.skills import SkillCatalog, SkillDefinition
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.proactive.content_turn import ContentTurnResult
from memopilot.proactive.mcp_sources import ProactiveEvent, ProactiveFetchResult
from memopilot.proactive.service import ProactiveService
from memopilot.proactive.store import ProactiveRepository

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)
SESSION = "feishu:chat-1"


@dataclass
class _Source:
    events: tuple[ProactiveEvent, ...] = ()
    calls: list[str] = field(default_factory=list)

    async def fetch(self):
        self.calls.append("fetch")
        return ProactiveFetchResult(self.events)

    async def ack(self, event_id: str, *, operation_id: str, ttl_hours: int):
        self.calls.append(f"ack:{event_id}:{operation_id}:{ttl_hours}")


class _Gateway:
    def __init__(self, repository: ProactiveRepository, source: _Source) -> None:
        self.repository = repository
        self.source = source

    async def collect(self, *, session_key: str, fetched_at: datetime):
        return self.repository.commit_fetch(
            session_key=session_key,
            source_id="source",
            result=await self.source.fetch(),
            fetched_at=fetched_at,
        )

    async def replay_pending_acknowledgements(self, *, now: datetime):
        for pending in self.repository.list_pending_acknowledgements(now):
            await self.source.ack(
                pending.source_event_id,
                operation_id=pending.ack_operation_id,
                ttl_hours=pending.ttl_hours,
            )
            self.repository.mark_acknowledged(pending.acknowledgement_id, acknowledged_at=now)


@dataclass
class _Decisions:
    values: list[ContentTurnResult] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    inputs: list[object] = field(default_factory=list)

    async def run(self, tick_input, **kwargs):
        del kwargs
        self.inputs.append(tick_input)
        self.calls.extend(
            f"alert:{event['event_id']}" for event in tick_input.alerts
        )
        if self.values:
            return self.values.pop(0)
        ids = frozenset(item.item_id for item in tick_input.contents)
        return ContentTurnResult("skip", discarded_item_ids=ids, reason="no_content")


@dataclass
class _Enqueuer:
    calls: list[dict[str, object]] = field(default_factory=list)

    def enqueue_drift(self, **kwargs):
        self.calls.append(kwargs)
        return True


def _event(event_id: str, kind: str, **payload) -> ProactiveEvent:
    return ProactiveEvent("source", event_id, kind, NOW.isoformat(), {"title": event_id, **payload})


def _skill() -> SkillDefinition:
    return SkillDefinition(
        name="background",
        description="后台任务",
        background_allowed=True,
        required_tools=(),
        content="执行后台任务",
        source="workspace",
        path=Path("background/SKILL.md"),
    )


def _service(
    tmp_path: Path,
    *,
    events: tuple[ProactiveEvent, ...] = (),
    decisions: _Decisions | None = None,
    skills: SkillCatalog | None = None,
    drift_enabled: bool = True,
    wake_policy=lambda _session, _now: False,
):
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    repository = ProactiveRepository(database)
    source = _Source(events)
    engine = decisions or _Decisions()
    enqueuer = _Enqueuer()
    service = ProactiveService(
        repository,
        _Gateway(repository, source),  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        skill_catalog=skills or SkillCatalog(),
        job_enqueuer=enqueuer,
        assert_current=lambda: None,
        drift_enabled=drift_enabled,
        drift_min_interval=timedelta(hours=3),
        wake_policy=wake_policy,
    )
    return service, repository, source, engine, enqueuer


@pytest.mark.asyncio
async def test_one_tick_batches_all_alerts_in_one_agent_tick(tmp_path: Path) -> None:
    decisions = _Decisions(
        [
            ContentTurnResult(
                "reply",
                "两条提醒",
                cited_item_ids=("source:a1", "source:a2"),
            )
        ]
    )
    service, repository, _, engine, _ = _service(
        tmp_path,
        events=(_event("a1", "alert"), _event("a2", "alert"), _event("c1", "content")),
        decisions=decisions,
    )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "send"
    assert engine.calls == ["alert:a1", "alert:a2"]
    assert len(repository.list_unconsumed(SESSION)) == 3


@pytest.mark.asyncio
async def test_confirmed_alert_is_consumed_and_ack_replayed_before_next_fetch(
    tmp_path: Path,
) -> None:
    decisions = _Decisions(
        [ContentTurnResult("reply", "提醒", cited_item_ids=("source:a1",))]
    )
    service, repository, source, _, _ = _service(
        tmp_path, events=(_event("a1", "alert"),), decisions=decisions
    )
    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)
    assert service.finalize_confirmed(outcome, confirmed_at=NOW)

    source.events = ()
    await service.execute("job-2", SESSION, "chat-1", 1, NOW + timedelta(seconds=1))

    assert source.calls[1].startswith("ack:a1:")
    assert source.calls[2] == "fetch"
    assert repository.list_unconsumed(SESSION) == ()


@pytest.mark.asyncio
async def test_unchanged_context_is_consumed_without_llm_or_source_ack(
    tmp_path: Path,
) -> None:
    service, repository, _, engine, _ = _service(
        tmp_path,
        events=(
            _event(
                "ctx",
                "context",
                presence="active",
                confidence=0.9,
                observed_at=NOW.isoformat(),
                expires_at=(NOW + timedelta(minutes=10)).isoformat(),
            ),
        ),
        drift_enabled=False,
    )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "quiet"
    assert engine.calls == []
    assert repository.list_unconsumed(SESSION) == ()
    assert repository.list_pending_acknowledgements(NOW) == ()


@pytest.mark.asyncio
async def test_context_is_raw_background_and_only_sends_when_fallback_is_open(
    tmp_path: Path,
) -> None:
    engine = _Decisions([ContentTurnResult("reply", "顺着刚才的话题聊聊")])
    service, repository, _, _, _ = _service(
        tmp_path,
        events=(
            _event(
                "ctx",
                "context",
                available=True,
                sleep_prob=0.2,
                custom_metric="kept",
            ),
        ),
        decisions=engine,
        wake_policy=lambda _session, _now: True,
    )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "send"
    assert outcome.trigger_kind == "context"
    tick_input = engine.inputs[0]
    assert tick_input.context_as_fallback_open is True
    assert tick_input.contexts[0]["custom_metric"] == "kept"
    assert repository.list_unconsumed(SESSION)[0].source_event_id == "ctx"
    assert service.finalize_confirmed(outcome, confirmed_at=NOW)
    assert repository.list_unconsumed(SESSION) == ()
    assert repository.list_pending_acknowledgements(NOW) == ()


@pytest.mark.asyncio
async def test_drift_enqueues_p3_when_background_skill_exists(tmp_path: Path) -> None:
    service, _, _, _, enqueuer = _service(tmp_path, skills=SkillCatalog((_skill(),)))

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "drift"
    assert enqueuer.calls[0]["priority"] == 3
