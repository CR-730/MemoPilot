"""旧原型 AgentTick 主动业务链的领域编排；外部发送仍由 Effect/Outbox 执行。"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from hashlib import sha1
from typing import Literal, Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid5

from memopilot.extensions.skills import SkillCatalog
from memopilot.proactive.content_turn import AgentTickInput, AgentTickResult
from memopilot.proactive.investigation import ContentCandidate
from memopilot.proactive.mcp_sources import ProactiveSourceGateway
from memopilot.proactive.store import (
    ProactiveDecisionRecord,
    ProactiveRepository,
    StoredProactiveEvent,
)

ProactiveAction = Literal["quiet", "send", "drift"]


@dataclass(frozen=True, slots=True)
class ProactiveOutcome:
    action: ProactiveAction
    session_key: str = ""
    trigger_kind: str = ""
    message: str = ""
    reason: str = ""
    decision_id: str | None = None
    effect_operation_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    source_events: tuple[tuple[str, str], ...] = ()
    drift_job_id: str | None = None
    decided_at: datetime | None = None
    delivery_key: str = ""


class AgentTickEngine(Protocol):
    async def run(
        self,
        tick_input: AgentTickInput,
        *,
        session_key: str,
        now: datetime,
        memory_text: str = "",
        proactive_context: str = "",
        recent_context: str = "",
    ) -> AgentTickResult: ...


class SemanticDeduper(Protocol):
    async def is_duplicate(
        self, new_message: str, recent_proactive: Sequence[str]
    ) -> tuple[bool, str]: ...


class ProactiveService:
    def __init__(
        self,
        repository: ProactiveRepository,
        source_gateway: ProactiveSourceGateway,
        agent_tick: AgentTickEngine,
        *,
        skill_catalog: SkillCatalog,
        assert_current: Callable[[], None],
        drift_min_interval: timedelta = timedelta(hours=3),
        drift_enabled: bool = True,
        context_probability: float = 0.3,
        random_value: Callable[[], float] = random.random,
        active_timezone: tzinfo,
        active_start_hour: int = 8,
        active_end_hour: int = 23,
        ordinary_daily_limit: int = 3,
        ordinary_cooldown: timedelta = timedelta(hours=2),
        alert_cooldown: timedelta = timedelta(minutes=30),
        message_deduper: SemanticDeduper | None = None,
        memory_text: Callable[[], str] = lambda: "",
        proactive_context: Callable[[], str] = lambda: "",
        recent_context: Callable[[str, datetime], str] = lambda _session, _now: "",
    ) -> None:
        if drift_min_interval < timedelta(0):
            raise ValueError("drift_min_interval 不能小于 0")
        if not 0 <= context_probability <= 1:
            raise ValueError("context_probability 必须在 0 到 1 之间")
        if not 0 <= active_start_hour < active_end_hour <= 24:
            raise ValueError("主动发送时段必须满足 0 <= start < end <= 24")
        if ordinary_daily_limit <= 0:
            raise ValueError("ordinary_daily_limit 必须大于 0")
        if ordinary_cooldown < timedelta(0) or alert_cooldown < timedelta(0):
            raise ValueError("主动发送冷却不能小于 0")
        self._repository = repository
        self._source_gateway = source_gateway
        self._agent_tick = agent_tick
        self._skill_catalog = skill_catalog
        self._assert_current = assert_current
        self._drift_min_interval = drift_min_interval
        self._drift_enabled = drift_enabled
        self._context_probability = context_probability
        self._random_value = random_value
        self._active_timezone = active_timezone
        self._active_start_hour = active_start_hour
        self._active_end_hour = active_end_hour
        self._ordinary_daily_limit = ordinary_daily_limit
        self._ordinary_cooldown = ordinary_cooldown
        self._alert_cooldown = alert_cooldown
        self._message_deduper = message_deduper
        self._memory_text = memory_text
        self._proactive_context = proactive_context
        self._recent_context = recent_context

    async def execute(
        self,
        job_id: str,
        session_key: str,
        chat_id: str,
        activity_version: int,
        now: datetime,
    ) -> ProactiveOutcome:
        del chat_id
        self._assert_current()
        await self._source_gateway.replay_pending_acknowledgements(now=now)
        await self._source_gateway.collect(session_key=session_key, fetched_at=now)
        pending = self._repository.find_pending_decision(session_key)
        if pending is not None:
            if pending.activity_version == activity_version:
                return self._recover_pending(pending, now=now)
            self._repository.cancel_stale_decision(
                pending.decision_id,
                current_activity_version=activity_version,
            )

        unread = self._repository.list_unconsumed(session_key)
        alerts = tuple(item for item in unread if item.kind == "alert")
        contexts = tuple(item for item in unread if item.kind == "context")
        contents = tuple(item for item in unread if item.kind == "content")[:5]

        if alerts:
            self._commit_context_snapshot(
                session_key, contexts, activity_version=activity_version, now=now
            )
            blocked = self._send_block_reason(session_key, "alert", now)
            if blocked:
                return ProactiveOutcome(
                    "quiet",
                    session_key=session_key,
                    trigger_kind="alert",
                    reason=blocked,
                    decided_at=now,
                )
            selected_alerts = alerts
            selected_contents: tuple[StoredProactiveEvent, ...] = ()
            selected_contexts: tuple[StoredProactiveEvent, ...] = ()
            fallback_open = False
        elif contents:
            self._commit_context_snapshot(
                session_key, contexts, activity_version=activity_version, now=now
            )
            blocked = self._send_block_reason(session_key, "content", now)
            if blocked:
                return ProactiveOutcome(
                    "quiet",
                    session_key=session_key,
                    trigger_kind="content",
                    reason=blocked,
                    decided_at=now,
                )
            selected_alerts = ()
            selected_contents = contents
            selected_contexts = ()
            fallback_open = False
        elif contexts and self._random_value() < self._context_probability:
            blocked = self._send_block_reason(session_key, "context", now)
            if blocked:
                return ProactiveOutcome(
                    "quiet",
                    session_key=session_key,
                    trigger_kind="context",
                    reason=blocked,
                    decided_at=now,
                )
            selected_alerts = ()
            selected_contents = ()
            selected_contexts = contexts
            fallback_open = True
        else:
            self._commit_context_snapshot(
                session_key, contexts, activity_version=activity_version, now=now
            )
            return self._handle_drift(job_id, session_key, now)

        self._assert_current()
        result = await self._agent_tick.run(
            AgentTickInput(
                alerts=tuple(_event_payload(item) for item in selected_alerts),
                contents=tuple(_content_candidate(item) for item in selected_contents),
                contexts=tuple(_context_payload(item) for item in selected_contexts),
                context_as_fallback_open=fallback_open,
            ),
            session_key=session_key,
            now=now,
            memory_text=str(self._memory_text() or ""),
            proactive_context=str(self._proactive_context() or ""),
            recent_context=str(self._recent_context(session_key, now) or ""),
        )
        self._assert_current()
        return await self._resolve_tick(
            result,
            session_key=session_key,
            activity_version=activity_version,
            now=now,
            alerts=selected_alerts,
            contents=selected_contents,
            contexts=selected_contexts,
            context_as_fallback_open=fallback_open,
        )

    async def _resolve_tick(
        self,
        result: AgentTickResult,
        *,
        session_key: str,
        activity_version: int,
        now: datetime,
        alerts: Sequence[StoredProactiveEvent],
        contents: Sequence[StoredProactiveEvent],
        contexts: Sequence[StoredProactiveEvent],
        context_as_fallback_open: bool,
    ) -> ProactiveOutcome:
        content_by_id = {_compound_id(item): item for item in contents}
        alert_by_id = {_compound_id(item): item for item in alerts}
        context_by_id = {_compound_id(item): item for item in contexts}
        known_ids = set(content_by_id) | set(alert_by_id) | set(context_by_id)
        classified = set(result.interesting_item_ids) | set(result.discarded_item_ids)
        if classified - set(content_by_id):
            raise ValueError("分类结果包含不属于本 tick 的候选")
        if set(result.interesting_item_ids) & set(result.discarded_item_ids):
            raise ValueError("条目不能同时标记 interesting 和 discarded")
        if set(result.cited_item_ids) - known_ids:
            raise ValueError("evidence 必须引用本 tick 的真实候选")
        if alerts and classified:
            raise ValueError("存在 Alert 时不能执行 Content 分类")
        if contents and not alerts and classified != set(content_by_id):
            raise ValueError("Content 路径必须且只能分类本 tick 的全部候选")
        if not set(result.cited_item_ids) <= set(result.interesting_item_ids) | set(alert_by_id):
            raise ValueError("Content 引用必须来自 interesting 条目，Alert 必须来自本 tick")

        discarded = tuple(
            item for item in contents if _compound_id(item) in result.discarded_item_ids
        )
        discarded_record = self._commit_discarded(
            session_key,
            discarded,
            result=result,
            activity_version=activity_version,
            now=now,
        )
        if result.action == "skip":
            self._commit_context_snapshot(
                session_key, contexts, activity_version=activity_version, now=now
            )
            if result.interesting_item_ids:
                return ProactiveOutcome(
                    "quiet",
                    session_key=session_key,
                    trigger_kind="content",
                    reason=result.reason,
                    decided_at=now,
                )
            if discarded_record is None:
                return ProactiveOutcome(
                    "quiet",
                    session_key=session_key,
                    reason=result.reason,
                    decided_at=now,
                )
            return ProactiveOutcome(
                "quiet",
                session_key=session_key,
                trigger_kind="content",
                reason=result.reason,
                decision_id=discarded_record.decision_id,
                evidence_ids=tuple(result.discarded_item_ids),
                source_events=discarded_record.source_events,
                decided_at=now,
            )

        if not result.message.strip():
            raise ValueError("AgentTick reply 必须包含非空消息")

        # Resolve 只把统一 AgentTickResult 投影为外层持久化记录；
        # Alert/Content/Context 不再各自拥有一套业务决策器。
        if alerts:
            selected = tuple(alerts)
            trigger_kind = "alert"
            action = "alert"
            evidence = result.cited_item_ids
            if set(evidence) != set(alert_by_id):
                raise ValueError("Alert 回复必须引用本 tick 的全部 Alert")
            ack_ttls: dict[str, int] = {}
        elif result.interesting_item_ids:
            selected = tuple(
                item for item in contents if _compound_id(item) in result.interesting_item_ids
            )
            trigger_kind = "content"
            action = "share"
            evidence = result.cited_item_ids
            ack_ttls = {
                item_id: (168 if item_id in result.cited_item_ids else 24)
                for item_id in result.interesting_item_ids
            }
        else:
            if not context_as_fallback_open or not contexts or result.cited_item_ids:
                raise ValueError("Context fallback 必须有 Context 快照且 evidence 为空")
            selected = tuple(contexts)
            trigger_kind = "context"
            action = "send_event"
            evidence = ()
            ack_ttls = {}

        if trigger_kind != "context":
            self._commit_context_snapshot(
                session_key, contexts, activity_version=activity_version, now=now
            )
        event_by_id = alert_by_id | content_by_id
        delivery_key = _build_delivery_key(result, event_by_id)
        if (
            self._repository.is_delivery_duplicate(
                session_key, delivery_key, window=timedelta(hours=24), now=now
            )
            or await self._is_semantic_duplicate(session_key, result.message)
        ):
            duplicate = self._commit_duplicate(
                session_key, selected, activity_version=activity_version, now=now
            )
            return ProactiveOutcome(
                "quiet",
                session_key=session_key,
                trigger_kind=trigger_kind,
                reason="already_sent_similar",
                decision_id=duplicate.decision_id,
                source_events=duplicate.source_events,
                decided_at=now,
            )
        stored = self._repository.create_decision(
            decision_id=_decision_id(session_key, trigger_kind, selected, activity_version),
            session_key=session_key,
            trigger_kind=trigger_kind,
            action=action,
            source_events=_source_refs(selected),
            activity_version=activity_version,
            decided_at=now,
            reason=result.reason or None,
            message=result.message,
            evidence=evidence,
            ack_ttl_hours=ack_ttls,
            delivery_key=delivery_key,
        )
        return self._send_outcome(stored, result, trigger_kind, session_key, now)

    async def _is_semantic_duplicate(self, session_key: str, message: str) -> bool:
        if self._message_deduper is None:
            return False
        recent = self._repository.list_recent_delivery_messages(session_key, limit=5)
        duplicate, _reason = await self._message_deduper.is_duplicate(message, recent)
        return duplicate

    def _commit_duplicate(
        self,
        session_key: str,
        events: Sequence[StoredProactiveEvent],
        *,
        activity_version: int,
        now: datetime,
    ) -> ProactiveDecisionRecord:
        kind = events[0].kind
        record = self._repository.create_decision(
            decision_id=_decision_id(session_key, f"{kind}-duplicate", events, activity_version),
            session_key=session_key,
            trigger_kind=kind,
            action="skip" if kind == "content" else "skip_event",
            source_events=_source_refs(events),
            activity_version=activity_version,
            decided_at=now,
            reason="already_sent_similar",
            ack_ttl_hours={_compound_id(item): 24 for item in events},
        )
        self._repository.commit_skip(record.decision_id, committed_at=now)
        return record

    def _commit_discarded(
        self,
        session_key: str,
        events: Sequence[StoredProactiveEvent],
        *,
        result: AgentTickResult,
        activity_version: int,
        now: datetime,
    ) -> ProactiveDecisionRecord | None:
        if not events:
            return None
        record = self._repository.create_decision(
            decision_id=_decision_id(session_key, "content-discarded", events, activity_version),
            session_key=session_key,
            trigger_kind="content",
            action="skip",
            source_events=_source_refs(events),
            activity_version=activity_version,
            decided_at=now,
            reason="not_interesting",
            evidence=tuple(result.discarded_item_ids),
            ack_ttl_hours={item: 720 for item in result.discarded_item_ids},
        )
        self._repository.commit_skip(record.decision_id, committed_at=now)
        return record

    def _commit_context_snapshot(
        self,
        session_key: str,
        events: Sequence[StoredProactiveEvent],
        *,
        activity_version: int,
        now: datetime,
    ) -> None:
        if not events:
            return
        for event in events:
            self._repository.save_context(
                source_id=event.source_id,
                payload=event.payload,
                fingerprint=_json_fingerprint(event.payload),
                observed_at=_parse_time(event.occurred_at) or now,
            )
        record = self._repository.create_decision(
            decision_id=_decision_id(session_key, "context-snapshot", events, activity_version),
            session_key=session_key,
            trigger_kind="context",
            action="skip_event",
            source_events=_source_refs(events),
            activity_version=activity_version,
            decided_at=now,
            reason="context_snapshot_consumed",
        )
        self._repository.commit_skip(record.decision_id, committed_at=now)

    def finalize_confirmed(
        self, outcome: ProactiveOutcome, *, confirmed_at: datetime | None = None
    ) -> bool:
        if outcome.action != "send" or outcome.decision_id is None:
            return False
        committed_at = confirmed_at or outcome.decided_at
        if committed_at is None:
            raise ValueError("confirmed outcome 缺少确认时间")
        operation_id = outcome.effect_operation_id
        finalized = self._repository.finalize_confirmed(
            outcome.decision_id,
            is_effect_confirmed=lambda value: operation_id is not None and value == operation_id,
            committed_at=committed_at,
        )
        if finalized and outcome.delivery_key:
            self._repository.mark_delivery(
                outcome.session_key,
                outcome.delivery_key,
                message=outcome.message,
                sent_at=committed_at,
            )
        return finalized

    def finalize_failed(
        self, outcome: ProactiveOutcome, *, failed_at: datetime | None = None
    ) -> bool:
        del failed_at
        if outcome.action != "send" or outcome.decision_id is None:
            return False
        return self._repository.mark_decision_failed(outcome.decision_id)

    def _send_block_reason(
        self,
        session_key: str,
        trigger_kind: str,
        now: datetime,
    ) -> str:
        if trigger_kind == "alert":
            last_alert = self._repository.last_confirmed_send(
                session_key, trigger_kinds=("alert",)
            )
            if last_alert is not None and now - last_alert < self._alert_cooldown:
                return "alert_cooldown"
            return ""
        local_now = now.astimezone(self._active_timezone)
        if not self._active_start_hour <= local_now.hour < self._active_end_hour:
            return "outside_active_window"
        local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        if (
            self._repository.count_confirmed_sends(
                session_key,
                trigger_kind=trigger_kind,
                since=local_midnight,
            )
            >= self._ordinary_daily_limit
        ):
            return "daily_limit"
        last_ordinary = self._repository.last_confirmed_send(
            session_key, trigger_kinds=("content", "context")
        )
        if (
            last_ordinary is not None
            and now - last_ordinary < self._ordinary_cooldown
        ):
            return "shared_cooldown"
        return ""

    def _handle_drift(
        self, job_id: str, session_key: str, now: datetime
    ) -> ProactiveOutcome:
        if not self._drift_enabled:
            return ProactiveOutcome("quiet", session_key=session_key, reason="drift_disabled")
        if not self._skill_catalog.background_candidates():
            return ProactiveOutcome("quiet", session_key=session_key, reason="drift_no_skill")
        last = self._repository.load_last_drift_at(session_key)
        if last is not None and now - last < self._drift_min_interval:
            return ProactiveOutcome("quiet", session_key=session_key, reason="drift_cooldown")
        self._assert_current()
        self._repository.mark_drift_started(
            session_key=session_key, job_id=job_id, started_at=now
        )
        return ProactiveOutcome(
            "drift", session_key=session_key, drift_job_id=job_id, decided_at=now
        )

    @staticmethod
    def _send_outcome(
        stored: ProactiveDecisionRecord,
        result: AgentTickResult,
        trigger_kind: str,
        session_key: str,
        now: datetime,
    ) -> ProactiveOutcome:
        return ProactiveOutcome(
            "send",
            session_key=session_key,
            trigger_kind=trigger_kind,
            message=result.message,
            reason=result.reason,
            decision_id=stored.decision_id,
            effect_operation_id=stored.effect_operation_id,
            evidence_ids=result.cited_item_ids,
            source_events=stored.source_events,
            decided_at=now,
            delivery_key=stored.delivery_key,
        )

    def _recover_pending(
        self, decision: ProactiveDecisionRecord, *, now: datetime
    ) -> ProactiveOutcome:
        if decision.action in {"share", "alert", "send_event"}:
            return ProactiveOutcome(
                "send",
                session_key=decision.session_key,
                trigger_kind=decision.trigger_kind,
                message=decision.message,
                reason=decision.reason,
                decision_id=decision.decision_id,
                effect_operation_id=decision.effect_operation_id,
                evidence_ids=decision.evidence,
                source_events=decision.source_events,
                decided_at=now,
                delivery_key=decision.delivery_key,
            )
        if decision.action in {"skip", "skip_event"}:
            self._repository.commit_skip(decision.decision_id, committed_at=now)
            return ProactiveOutcome(
                "quiet", session_key=decision.session_key, reason=decision.reason, decided_at=now
            )
        raise ValueError(f"无法恢复未知主动决策动作: {decision.action}")


def _content_candidate(event: StoredProactiveEvent) -> ContentCandidate:
    return ContentCandidate(
        item_id=_compound_id(event),
        title=str(event.payload.get("title") or ""),
        source_id=event.source_id,
        published_at=str(event.payload.get("published_at") or event.occurred_at),
        payload=event.payload,
    )


def _event_payload(event: StoredProactiveEvent) -> dict[str, object]:
    return {
        "source_id": event.source_id,
        "event_id": event.source_event_id,
        "occurred_at": event.occurred_at,
        **event.payload,
    }


def _context_payload(event: StoredProactiveEvent) -> dict[str, object]:
    return {"_source": event.source_id, **event.payload}


def _compound_id(event: StoredProactiveEvent) -> str:
    return f"{event.source_id}:{event.source_event_id}"


def _source_refs(events: Sequence[StoredProactiveEvent]) -> tuple[tuple[str, str], ...]:
    return tuple((item.source_id, item.source_event_id) for item in events)


def _decision_id(
    session_key: str,
    kind: str,
    events: Sequence[StoredProactiveEvent],
    activity_version: int,
) -> str:
    identity = ":".join(item.reservoir_id for item in events)
    return _stable_id("proactive-decision", f"{session_key}:{activity_version}:{kind}:{identity}")


def _stable_id(kind: str, identity: str) -> str:
    return f"{kind}-{uuid5(NAMESPACE_URL, f'memopilot:{kind}:{identity}').hex}"


def _json_fingerprint(payload: dict[str, object]) -> str:
    from hashlib import sha256

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _build_delivery_key(
    result: AgentTickResult,
    event_by_id: dict[str, StoredProactiveEvent],
) -> str:
    refs: list[str] = []
    for item_id in sorted(set(result.cited_item_ids)):
        event = event_by_id.get(item_id)
        if event is None:
            refs.append(f"id:{item_id}")
            continue
        url = _normalize_delivery_url(str(event.payload.get("url") or ""))
        if url:
            refs.append(f"url:{url}")
            continue
        source = (
            str(event.payload.get("source_name") or event.payload.get("source") or event.source_id)
            .strip()
            .lower()
        )
        title = str(event.payload.get("title") or "").strip().lower()
        refs.append(f"title:{source}|{title}" if title else f"id:{item_id}")
    if refs and any(not ref.startswith("id:") for ref in refs):
        raw = json.dumps(sorted(set(refs)))
    elif result.cited_item_ids:
        raw = json.dumps(sorted(result.cited_item_ids))
    else:
        raw = result.message[:500]
    return sha1(raw.encode()).hexdigest()[:16]


def _normalize_delivery_url(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    parts = urlsplit(text)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _parse_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = ["AgentTickEngine", "ProactiveOutcome", "ProactiveService"]
