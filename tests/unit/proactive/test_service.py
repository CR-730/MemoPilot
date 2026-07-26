from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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


def _commit_confirmed_send(
    repository: ProactiveRepository,
    *,
    event_id: str,
    kind: str,
    committed_at: datetime,
) -> None:
    event = ProactiveEvent(
        "seed",
        event_id,
        kind,
        committed_at.isoformat(),
        {"title": event_id},
    )
    repository.commit_fetch(
        session_key=SESSION,
        source_id="seed",
        result=ProactiveFetchResult((event,)),
        fetched_at=committed_at,
    )
    decision = repository.create_decision(
        decision_id=f"decision-{event_id}",
        session_key=SESSION,
        trigger_kind=kind,
        action={"alert": "alert", "content": "share", "context": "send_event"}[kind],
        source_events=(("seed", event_id),),
        activity_version=1,
        decided_at=committed_at,
        message=event_id,
    )
    assert repository.finalize_confirmed(
        decision.decision_id,
        is_effect_confirmed=lambda _operation_id: True,
        committed_at=committed_at,
    )


def _service(
    tmp_path: Path,
    *,
    events: tuple[ProactiveEvent, ...] = (),
    decisions: _Decisions | None = None,
    skills: SkillCatalog | None = None,
    drift_enabled: bool = True,
    random_value=lambda: 1.0,
    assert_current=lambda: None,
):
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    repository = ProactiveRepository(database)
    source = _Source(events)
    engine = decisions or _Decisions()
    service = ProactiveService(
        repository,
        _Gateway(repository, source),  # type: ignore[arg-type]
        engine,  # type: ignore[arg-type]
        skill_catalog=skills or SkillCatalog(),
        assert_current=assert_current,
        drift_enabled=drift_enabled,
        drift_min_interval=timedelta(hours=3),
        context_probability=0.3,
        random_value=random_value,
        active_timezone=ZoneInfo("Asia/Shanghai"),
    )
    return service, repository, source, engine


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
    service, repository, _, engine = _service(
        tmp_path,
        events=(_event("a1", "alert"), _event("a2", "alert"), _event("c1", "content")),
        decisions=decisions,
    )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "send"
    assert engine.calls == ["alert:a1", "alert:a2"]
    assert engine.inputs[0].contents == ()
    assert engine.inputs[0].contexts == ()
    assert len(repository.list_unconsumed(SESSION)) == 3


@pytest.mark.asyncio
async def test_confirmed_alert_is_consumed_and_ack_replayed_before_next_fetch(
    tmp_path: Path,
) -> None:
    decisions = _Decisions(
        [ContentTurnResult("reply", "提醒", cited_item_ids=("source:a1",))]
    )
    service, repository, source, _ = _service(
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
    service, repository, _, engine = _service(
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
    service, repository, _, _ = _service(
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
        random_value=lambda: 0.29,
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
async def test_content_route_excludes_context_and_does_not_evaluate_context_gate(
    tmp_path: Path,
) -> None:
    gate_calls = 0

    def random_value() -> float:
        nonlocal gate_calls
        gate_calls += 1
        return 0.0

    engine = _Decisions(
        [
            ContentTurnResult(
                "skip",
                discarded_item_ids=frozenset({"source:content"}),
                reason="no_content",
            )
        ]
    )
    service, _, _, _ = _service(
        tmp_path,
        events=(_event("content", "content"), _event("context", "context")),
        decisions=engine,
        random_value=random_value,
        drift_enabled=False,
    )

    await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert engine.inputs[0].alerts == ()
    assert len(engine.inputs[0].contents) == 1
    assert engine.inputs[0].contexts == ()
    assert gate_calls == 0


@pytest.mark.asyncio
async def test_context_gate_opens_only_below_thirty_percent(tmp_path: Path) -> None:
    (tmp_path / "closed").mkdir()
    (tmp_path / "opened").mkdir()
    closed_engine = _Decisions()
    closed, _, _, _ = _service(
        tmp_path / "closed",
        events=(_event("ctx", "context"),),
        decisions=closed_engine,
        random_value=lambda: 0.3,
        drift_enabled=False,
    )
    opened_engine = _Decisions([ContentTurnResult("reply", "聊聊")])
    opened, _, _, _ = _service(
        tmp_path / "opened",
        events=(_event("ctx", "context"),),
        decisions=opened_engine,
        random_value=lambda: 0.299,
        drift_enabled=False,
    )

    closed_outcome = await closed.execute("closed", SESSION, "chat-1", 1, NOW)
    opened_outcome = await opened.execute("opened", SESSION, "chat-1", 1, NOW)

    assert closed_outcome.action == "quiet"
    assert closed_engine.inputs == []
    assert opened_outcome.action == "send"
    assert len(opened_engine.inputs) == 1


@pytest.mark.asyncio
async def test_content_and_context_have_independent_daily_limits(tmp_path: Path) -> None:
    service, repository, _, engine = _service(
        tmp_path,
        events=(_event("new-content", "content"),),
        decisions=_Decisions(),
        drift_enabled=False,
    )
    for index in range(3):
        _commit_confirmed_send(
            repository,
            event_id=f"content-{index}",
            kind="content",
            committed_at=NOW - timedelta(hours=6 - index * 2),
        )
    for index in range(2):
        _commit_confirmed_send(
            repository,
            event_id=f"context-{index}",
            kind="context",
            committed_at=NOW - timedelta(hours=5 - index * 2),
        )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "quiet"
    assert outcome.reason == "daily_limit"
    assert engine.inputs == []
    assert any(
        item.source_event_id == "new-content"
        for item in repository.list_unconsumed(SESSION)
    )


@pytest.mark.asyncio
async def test_content_and_context_share_two_hour_cooldown(tmp_path: Path) -> None:
    engine = _Decisions([ContentTurnResult("reply", "状态提醒")])
    service, repository, _, _ = _service(
        tmp_path,
        events=(_event("new-context", "context"),),
        decisions=engine,
        random_value=lambda: 0.0,
        drift_enabled=False,
    )
    _commit_confirmed_send(
        repository,
        event_id="recent-content",
        kind="content",
        committed_at=NOW - timedelta(hours=1),
    )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "quiet"
    assert outcome.reason == "shared_cooldown"
    assert engine.inputs == []


@pytest.mark.asyncio
async def test_ordinary_push_respects_configured_active_window(tmp_path: Path) -> None:
    local_23 = datetime(2026, 7, 21, 23, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    service, _, _, engine = _service(
        tmp_path,
        events=(_event("content", "content"),),
        decisions=_Decisions(),
        drift_enabled=False,
    )

    outcome = await service.execute(
        "job", SESSION, "chat-1", 1, local_23.astimezone(UTC)
    )

    assert outcome.action == "quiet"
    assert outcome.reason == "outside_active_window"
    assert engine.inputs == []


@pytest.mark.asyncio
async def test_ordinary_push_allows_active_window_start_boundary(tmp_path: Path) -> None:
    local_08 = datetime(2026, 7, 21, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    engine = _Decisions()
    service, _, _, _ = _service(
        tmp_path,
        events=(_event("content", "content"),),
        decisions=engine,
        drift_enabled=False,
    )

    await service.execute("job", SESSION, "chat-1", 1, local_08.astimezone(UTC))

    assert len(engine.inputs) == 1


@pytest.mark.asyncio
async def test_alert_anti_storm_keeps_new_alert_for_next_tick(tmp_path: Path) -> None:
    service, repository, _, engine = _service(
        tmp_path,
        events=(_event("new-alert", "alert"),),
        decisions=_Decisions(
            [ContentTurnResult("reply", "提醒", cited_item_ids=("source:new-alert",))]
        ),
    )
    _commit_confirmed_send(
        repository,
        event_id="recent-alert",
        kind="alert",
        committed_at=NOW - timedelta(minutes=10),
    )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "quiet"
    assert outcome.reason == "alert_cooldown"
    assert engine.inputs == []
    assert any(
        item.source_event_id == "new-alert"
        for item in repository.list_unconsumed(SESSION)
    )


@pytest.mark.asyncio
async def test_execution_ownership_is_checked_before_external_gateway_calls(
    tmp_path: Path,
) -> None:
    def lost() -> None:
        raise RuntimeError("lost")

    service, _, source, _ = _service(tmp_path, assert_current=lost)

    with pytest.raises(RuntimeError, match="lost"):
        await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert source.calls == []


@pytest.mark.asyncio
async def test_drift_runs_inside_current_proactive_job_without_p3_enqueue(
    tmp_path: Path,
) -> None:
    service, repository, _, _ = _service(
        tmp_path, skills=SkillCatalog((_skill(),))
    )

    outcome = await service.execute("job", SESSION, "chat-1", 1, NOW)

    assert outcome.action == "drift"
    assert outcome.drift_job_id == "job"
    assert repository.load_last_drift_at(SESSION) == NOW


@pytest.mark.asyncio
async def test_drift_ignores_ordinary_send_budget_and_active_window(
    tmp_path: Path,
) -> None:
    local_02 = datetime(2026, 7, 22, 2, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    now = local_02.astimezone(UTC)
    service, repository, _, _ = _service(
        tmp_path, skills=SkillCatalog((_skill(),))
    )
    for index in range(3):
        _commit_confirmed_send(
            repository,
            event_id=f"content-{index}",
            kind="content",
            committed_at=now - timedelta(hours=5 - index),
        )

    outcome = await service.execute("job", SESSION, "chat-1", 1, now)

    assert outcome.action == "drift"
