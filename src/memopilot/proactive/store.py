"""Proactive 事件、决策和可恢复状态的 SQLite 事实源。"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from memopilot.persistence.migrations import connect_database
from memopilot.proactive.mcp_sources import (
    ProactiveFetchResult,
    stable_ack_operation_id,
)


@dataclass(frozen=True, slots=True)
class StoredProactiveEvent:
    reservoir_id: str
    source_id: str
    source_event_id: str
    session_key: str
    kind: str
    occurred_at: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProactiveDecisionRecord:
    decision_id: str
    session_key: str
    trigger_kind: str
    action: str
    source_events: tuple[tuple[str, str], ...]
    effect_operation_id: str | None
    activity_version: int
    state: str
    message: str = ""
    evidence: tuple[str, ...] = ()
    reason: str = ""
    delivery_key: str = ""


@dataclass(frozen=True, slots=True)
class PendingAcknowledgement:
    acknowledgement_id: str
    source_id: str
    source_event_id: str
    ack_operation_id: str
    attempts: int
    next_attempt_at: str
    ttl_hours: int


@dataclass(frozen=True, slots=True)
class StoredContext:
    source_id: str
    payload: dict[str, Any]
    fingerprint: str
    observed_at: str
    updated_at: str


def stable_proactive_effect_operation_id(decision_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"memopilot:proactive-effect:{decision_id}"))


class ProactiveRepository:
    """每个公开方法自带短事务，允许 Scheduler/Worker 分进程访问。"""

    def __init__(self, database: Path) -> None:
        self._database = Path(database)

    def commit_fetch(
        self,
        *,
        session_key: str,
        source_id: str,
        result: ProactiveFetchResult,
        fetched_at: datetime,
    ) -> int:
        """按 Source Event ID 幂等持久化本批事件。"""
        fetched = _utc_iso(fetched_at)
        serialized = [
            (
                str(
                    uuid5(
                        NAMESPACE_URL,
                        f"memopilot:proactive-event:{source_id}:{event.event_id}",
                    )
                ),
                source_id,
                event.event_id,
                session_key,
                event.kind,
                event.occurred_at,
                json.dumps(event.payload, ensure_ascii=False, sort_keys=True),
                fetched,
            )
            for event in result.events
        ]
        if any(event.source_id != source_id for event in result.events):
            raise ValueError("fetch 结果的 source_id 与注册 Source 不一致")
        with connect_database(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                inserted = 0
                for item in serialized:
                    existing = connection.execute(
                        """
                        SELECT e.reservoir_id, e.session_key, e.kind, e.occurred_at,
                               e.payload_json, e.fetched_at, e.consumed_at,
                               e.consume_reason, a.ack_operation_id, a.state,
                               a.ttl_hours, a.acknowledged_at
                        FROM source_events e
                        LEFT JOIN pending_acknowledgements a
                          ON a.source_id = e.source_id
                         AND a.source_event_id = e.source_event_id
                        WHERE e.source_id = ? AND e.source_event_id = ?
                        """,
                        (item[1], item[2]),
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            """
                            INSERT INTO source_events(
                                reservoir_id, source_id, source_event_id, session_key,
                                kind, occurred_at, payload_json, fetched_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            item,
                        )
                        inserted += 1
                        continue
                    if not _content_occurrence_expired(existing, fetched_at):
                        continue
                    archived_at = _utc_iso(fetched_at)
                    archive_id = str(
                        uuid5(
                            NAMESPACE_URL,
                            f"memopilot:proactive-event-archive:{existing[0]}",
                        )
                    )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO source_event_history(
                            archive_id, reservoir_id, source_id, source_event_id,
                            session_key, kind, occurred_at, payload_json, fetched_at,
                            consumed_at, consume_reason, ack_operation_id, ack_state,
                            ack_ttl_hours, acknowledged_at, archived_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            archive_id,
                            existing[0],
                            item[1],
                            item[2],
                            existing[1],
                            existing[2],
                            existing[3],
                            existing[4],
                            existing[5],
                            existing[6],
                            existing[7],
                            existing[8],
                            existing[9],
                            existing[10],
                            existing[11],
                            archived_at,
                        ),
                    )
                    connection.execute(
                        "DELETE FROM pending_acknowledgements "
                        "WHERE source_id = ? AND source_event_id = ?",
                        (item[1], item[2]),
                    )
                    occurrence_id = str(
                        uuid5(
                            NAMESPACE_URL,
                            f"memopilot:proactive-event-occurrence:{item[1]}:{item[2]}:{item[7]}",
                        )
                    )
                    connection.execute(
                        """
                        UPDATE source_events
                        SET reservoir_id = ?, session_key = ?, kind = ?, occurred_at = ?,
                            payload_json = ?, fetched_at = ?, consumed_at = NULL,
                            consume_reason = NULL
                        WHERE source_id = ? AND source_event_id = ?
                        """,
                        (
                            occurrence_id,
                            item[3],
                            item[4],
                            item[5],
                            item[6],
                            item[7],
                            item[1],
                            item[2],
                        ),
                    )
                    inserted += 1
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return inserted

    def list_unconsumed(
        self, session_key: str, *, limit: int | None = None
    ) -> tuple[StoredProactiveEvent, ...]:
        query = """
            SELECT reservoir_id, source_id, source_event_id, session_key, kind,
                   occurred_at, payload_json
            FROM source_events
            WHERE session_key = ? AND consumed_at IS NULL
            ORDER BY CASE kind
                WHEN 'alert' THEN 0 WHEN 'context' THEN 1 ELSE 2 END,
                CASE WHEN kind = 'alert' THEN source_id END ASC,
                CASE WHEN kind = 'alert' THEN occurred_at END DESC,
                occurred_at, fetched_at, source_id, source_event_id
        """
        parameters: list[object] = [session_key]
        if limit is not None:
            if limit <= 0:
                return ()
            query += " LIMIT ?"
            parameters.append(limit)
        with connect_database(self._database) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            StoredProactiveEvent(
                reservoir_id=str(row[0]),
                source_id=str(row[1]),
                source_event_id=str(row[2]),
                session_key=str(row[3]),
                kind=str(row[4]),
                occurred_at=str(row[5]),
                payload=json.loads(str(row[6])),
            )
            for row in rows
        )

    def record_observation(
        self,
        *,
        session_key: str,
        kind: str,
        subject_id: str,
        trigger: dict[str, Any],
        candidates: Sequence[dict[str, Any]],
        llm_input: Sequence[dict[str, Any]],
        created_at: datetime,
    ) -> str:
        observation_id = f"proactive-observation-{uuid4().hex}"
        with connect_database(self._database) as connection:
            connection.execute(
                """
                INSERT INTO proactive_observations(
                    observation_id, session_key, kind, subject_id, trigger_json,
                    candidates_json, llm_input_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    session_key,
                    kind,
                    subject_id,
                    json.dumps(trigger, ensure_ascii=False, default=str),
                    json.dumps(tuple(candidates), ensure_ascii=False, default=str),
                    json.dumps(tuple(llm_input), ensure_ascii=False, default=str),
                    _utc_iso(created_at),
                ),
            )
        return observation_id

    def list_observations(self, session_key: str) -> tuple[dict[str, Any], ...]:
        return self._list_audit("proactive_observations", session_key, "created_at")

    def list_drift_history(self, session_key: str) -> tuple[dict[str, Any], ...]:
        return self._list_audit("drift_history", session_key, "created_at")

    def load_last_drift_at(self, session_key: str) -> datetime | None:
        with connect_database(self._database) as connection:
            row = connection.execute(
                "SELECT last_drift_at FROM drift_state WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        if row is None or not row[0]:
            return None
        return _parse_time(row[0])

    def mark_drift_started(
        self,
        *,
        session_key: str,
        job_id: str,
        started_at: datetime,
    ) -> None:
        """记录当前主动 Job 已进入 Drift；沿用旧表保持数据库兼容。"""
        timestamp = _utc_iso(started_at)
        drift_id = str(uuid5(NAMESPACE_URL, f"memopilot:drift:{session_key}:{job_id}"))
        with connect_database(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO drift_history(
                        drift_id, session_key, skill_name, drive_before, threshold,
                        outcome, job_id, reason, created_at, updated_at, trace_json
                    ) VALUES (
                        ?, ?, NULL, 0, 0, 'running', ?,
                        'direct_proactive_fallback', ?, ?, '{}'
                    )
                    """,
                    (drift_id, session_key, job_id, timestamp, timestamp),
                )
                connection.execute(
                    """
                    INSERT INTO drift_state(
                        session_key, drive, threshold, updated_at, last_drift_at,
                        fingerprint, repeat_count
                    ) VALUES (?, 0, 0, ?, ?, '', 0)
                    ON CONFLICT(session_key) DO UPDATE SET
                        drive = 0,
                        threshold = 0,
                        updated_at = excluded.updated_at,
                        last_drift_at = excluded.last_drift_at,
                        fingerprint = '',
                        repeat_count = 0
                    """,
                    (session_key, timestamp, timestamp),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def complete_drift(
        self,
        *,
        session_key: str,
        job_id: str,
        skill_name: str,
        outcome: str,
        result: dict[str, str],
        completed_at: datetime,
    ) -> None:
        if outcome not in {"succeeded", "failed", "cancelled"}:
            raise ValueError(f"无效 Drift outcome: {outcome}")
        timestamp = _utc_iso(completed_at)
        reason = str(result.get("message_result") or "unfinished")
        trace = json.dumps(result, ensure_ascii=False, sort_keys=True)
        with connect_database(self._database) as connection:
            changed = connection.execute(
                """
                UPDATE drift_history
                SET skill_name = ?, outcome = ?, reason = ?,
                    updated_at = ?, trace_json = ?
                WHERE session_key = ? AND job_id = ? AND outcome = 'running'
                """,
                (
                    skill_name,
                    outcome,
                    reason,
                    timestamp,
                    trace,
                    session_key,
                    job_id,
                ),
            ).rowcount
        if changed != 1:
            raise KeyError(f"找不到运行中的 Drift: {session_key}/{job_id}")

    def _list_audit(
        self, table: str, session_key: str, order_column: str
    ) -> tuple[dict[str, Any], ...]:
        with connect_database(self._database) as connection:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE session_key = ? ORDER BY {order_column}",
                (session_key,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def create_decision(
        self,
        *,
        decision_id: str,
        session_key: str,
        trigger_kind: str,
        action: str,
        source_events: Sequence[tuple[str, str]],
        activity_version: int,
        decided_at: datetime,
        reason: str | None = None,
        message: str = "",
        evidence: Sequence[str] = (),
        ack_ttl_hours: dict[str, int] | None = None,
        delivery_key: str = "",
    ) -> ProactiveDecisionRecord:
        selected = tuple((str(source), str(event)) for source, event in source_events)
        if not selected:
            raise ValueError("Decision 必须至少引用一个 Source Event")
        effect_operation_id = (
            stable_proactive_effect_operation_id(decision_id)
            if action in {"share", "alert", "send_event"}
            else None
        )
        normalized_ack_ttls = {str(key): int(value) for key, value in (ack_ttl_hours or {}).items()}
        selected_keys = {f"{source}:{event}" for source, event in selected}
        if set(normalized_ack_ttls) - selected_keys:
            raise ValueError("ACK TTL 只能引用当前 Decision 的 Source Event")
        if any(value <= 0 for value in normalized_ack_ttls.values()):
            raise ValueError("ACK TTL 必须大于 0")
        decision_payload = json.dumps(
            {
                "message": str(message),
                "evidence": [str(item) for item in evidence],
                "reason": str(reason or ""),
                "ack_ttl_hours": normalized_ack_ttls,
                "delivery_key": str(delivery_key),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with connect_database(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for source_id, source_event_id in selected:
                    event = connection.execute(
                        """
                        SELECT session_key, kind, consumed_at FROM source_events
                        WHERE source_id = ? AND source_event_id = ?
                        """,
                        (source_id, source_event_id),
                    ).fetchone()
                    if event is None or str(event[0]) != session_key:
                        raise ValueError("Decision 只能引用当前会话已持久化的事件")
                    if str(event[1]) != trigger_kind:
                        raise ValueError("Decision trigger 类型必须与引用事件类型一致")
                    if event[2] is not None:
                        raise ValueError("Decision 不能重复引用已消费事件")
                serialized_events = json.dumps(selected, ensure_ascii=False)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO proactive_decisions(
                        decision_id, session_key, trigger_kind, action,
                        source_events_json, effect_operation_id, activity_version,
                        state, reason, decided_at, decision_payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        decision_id,
                        session_key,
                        trigger_kind,
                        action,
                        serialized_events,
                        effect_operation_id,
                        activity_version,
                        reason,
                        _utc_iso(decided_at),
                        decision_payload,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT session_key, trigger_kind, action, source_events_json,
                           effect_operation_id, activity_version, state,
                           decision_payload_json
                    FROM proactive_decisions WHERE decision_id = ?
                    """,
                    (decision_id,),
                ).fetchone()
                expected = (
                    session_key,
                    trigger_kind,
                    action,
                    serialized_events,
                    effect_operation_id,
                    activity_version,
                )
                if row is None or tuple(row[:6]) != expected:
                    raise ValueError("decision_id 已被不同参数占用")
                if str(row[7]) != decision_payload:
                    raise ValueError("decision_id 已被不同恢复 Payload 占用")
                state = str(row[6])
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return ProactiveDecisionRecord(
            decision_id=decision_id,
            session_key=session_key,
            trigger_kind=trigger_kind,
            action=action,
            source_events=selected,
            effect_operation_id=effect_operation_id,
            activity_version=activity_version,
            state=state,
            message=str(message),
            evidence=tuple(str(item) for item in evidence),
            reason=str(reason or ""),
            delivery_key=str(delivery_key),
        )

    def is_delivery_duplicate(
        self,
        session_key: str,
        delivery_key: str,
        *,
        window: timedelta,
        now: datetime,
    ) -> bool:
        cutoff = _utc_iso(now - max(window, timedelta(hours=1)))
        with connect_database(self._database) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM proactive_deliveries
                WHERE session_key = ? AND delivery_key = ? AND sent_at >= ?
                """,
                (session_key, delivery_key, cutoff),
            ).fetchone()
        return row is not None

    def mark_delivery(
        self,
        session_key: str,
        delivery_key: str,
        *,
        message: str,
        sent_at: datetime,
    ) -> None:
        with connect_database(self._database) as connection:
            connection.execute(
                """
                INSERT INTO proactive_deliveries(session_key, delivery_key, message, sent_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_key, delivery_key)
                DO UPDATE SET message = excluded.message, sent_at = excluded.sent_at
                """,
                (session_key, delivery_key, message, _utc_iso(sent_at)),
            )

    def list_recent_delivery_messages(self, session_key: str, *, limit: int = 5) -> tuple[str, ...]:
        with connect_database(self._database) as connection:
            rows = connection.execute(
                """
                SELECT message FROM proactive_deliveries
                WHERE session_key = ?
                ORDER BY sent_at DESC
                LIMIT ?
                """,
                (session_key, max(1, limit)),
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def count_confirmed_sends(
        self,
        session_key: str,
        *,
        trigger_kind: str,
        since: datetime,
    ) -> int:
        with connect_database(self._database) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM proactive_decisions
                WHERE session_key = ?
                  AND trigger_kind = ?
                  AND action IN ('share', 'alert', 'send_event')
                  AND state = 'committed'
                  AND committed_at >= ?
                """,
                (session_key, trigger_kind, _utc_iso(since)),
            ).fetchone()
        return int(row[0])

    def last_confirmed_send(
        self,
        session_key: str,
        *,
        trigger_kinds: Sequence[str],
    ) -> datetime | None:
        kinds = tuple(dict.fromkeys(str(item) for item in trigger_kinds if str(item)))
        if not kinds:
            return None
        placeholders = ", ".join("?" for _ in kinds)
        with connect_database(self._database) as connection:
            row = connection.execute(
                f"""
                SELECT MAX(committed_at) FROM proactive_decisions
                WHERE session_key = ?
                  AND trigger_kind IN ({placeholders})
                  AND action IN ('share', 'alert', 'send_event')
                  AND state = 'committed'
                """,
                (session_key, *kinds),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return _parse_time(row[0])

    def commit_skip(self, decision_id: str, *, committed_at: datetime) -> None:
        decision = self._load_decision(decision_id)
        if decision.action not in {"skip", "skip_event"}:
            raise ValueError(f"{decision.action} 决策必须等待 Effect confirmed，不能按 skip 提交")
        self._commit_decision(decision_id, committed_at=committed_at)

    def finalize_confirmed(
        self,
        decision_id: str,
        *,
        is_effect_confirmed: Callable[[str], bool],
        committed_at: datetime,
    ) -> bool:
        decision = self._load_decision(decision_id)
        operation_id = decision.effect_operation_id
        if operation_id is None or not is_effect_confirmed(operation_id):
            return False
        self._commit_decision(decision_id, committed_at=committed_at)
        return True

    def get_decision(self, decision_id: str) -> ProactiveDecisionRecord:
        decision = self.find_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        return decision

    def find_decision(self, decision_id: str) -> ProactiveDecisionRecord | None:
        with connect_database(self._database) as connection:
            row = connection.execute(
                """
                SELECT decision_id, session_key, trigger_kind, action,
                       source_events_json, effect_operation_id, activity_version, state,
                       decision_payload_json
                FROM proactive_decisions WHERE decision_id = ?
                """,
                (decision_id,),
            ).fetchone()
        return None if row is None else _proactive_decision(row)

    def find_pending_decision(
        self,
        session_key: str,
        *,
        trigger_kind: str | None = None,
    ) -> ProactiveDecisionRecord | None:
        condition = " AND trigger_kind = ?" if trigger_kind is not None else ""
        parameters: tuple[object, ...] = (
            (session_key, trigger_kind) if trigger_kind is not None else (session_key,)
        )
        with connect_database(self._database) as connection:
            row = connection.execute(
                """
                SELECT decision_id, session_key, trigger_kind, action,
                       source_events_json, effect_operation_id, activity_version, state,
                       decision_payload_json
                FROM proactive_decisions
                WHERE session_key = ? AND state = 'pending'
                """
                + condition
                + " ORDER BY decided_at, decision_id LIMIT 1",
                parameters,
            ).fetchone()
        return None if row is None else _proactive_decision(row)

    def cancel_stale_decision(
        self,
        decision_id: str,
        *,
        current_activity_version: int,
    ) -> bool:
        """仅作废旧 activity 的 pending 决策，不消费其 Source Event。"""
        with connect_database(self._database) as connection:
            changed = connection.execute(
                """
                UPDATE proactive_decisions
                SET state = 'cancelled'
                WHERE decision_id = ? AND state = 'pending'
                  AND activity_version <> ?
                """,
                (decision_id, current_activity_version),
            ).rowcount
        return bool(changed)

    def mark_decision_failed(self, decision_id: str) -> bool:
        """终止已知未产生外部副作用的发送决策，不消费 Source Event。"""
        with connect_database(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT action, state FROM proactive_decisions WHERE decision_id = ?",
                    (decision_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(decision_id)
                if str(row[0]) not in {"share", "alert", "send_event"}:
                    raise ValueError("只有发送型 Decision 可以标记为已知失败")
                state = str(row[1])
                if state == "pending":
                    connection.execute(
                        "UPDATE proactive_decisions SET state = 'failed' WHERE decision_id = ?",
                        (decision_id,),
                    )
                    state = "failed"
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return state == "failed"

    def _load_decision(self, decision_id: str) -> ProactiveDecisionRecord:
        return self.get_decision(decision_id)

    def _commit_decision(self, decision_id: str, *, committed_at: datetime) -> None:
        committed = _utc_iso(committed_at)
        with connect_database(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT source_events_json, state, trigger_kind, decision_payload_json
                    FROM proactive_decisions WHERE decision_id = ?
                    """,
                    (decision_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(decision_id)
                selected = tuple(tuple(item) for item in json.loads(str(row[0])))
                trigger_kind = str(row[2])
                payload = json.loads(str(row[3]))
                ack_ttl_hours = {
                    str(key): int(value) for key, value in payload.get("ack_ttl_hours", {}).items()
                }
                for source_id, source_event_id in selected:
                    connection.execute(
                        """
                        UPDATE source_events
                        SET consumed_at = COALESCE(consumed_at, ?),
                            consume_reason = COALESCE(consume_reason, ?)
                        WHERE source_id = ? AND source_event_id = ?
                        """,
                        (committed, decision_id, source_id, source_event_id),
                    )
                    if trigger_kind != "context":
                        acknowledgement_id = str(
                            uuid5(
                                NAMESPACE_URL,
                                f"memopilot:proactive-ack-row:{source_id}:{source_event_id}",
                            )
                        )
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO pending_acknowledgements(
                                acknowledgement_id, source_id, source_event_id,
                                ack_operation_id, state, attempts, next_attempt_at,
                                created_at, updated_at, ttl_hours
                            ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                            """,
                            (
                                acknowledgement_id,
                                source_id,
                                source_event_id,
                                stable_ack_operation_id(source_id, source_event_id),
                                committed,
                                committed,
                                committed,
                                ack_ttl_hours.get(f"{source_id}:{source_event_id}", 24),
                            ),
                        )
                connection.execute(
                    """
                    UPDATE proactive_decisions
                    SET state = 'committed', committed_at = COALESCE(committed_at, ?)
                    WHERE decision_id = ?
                    """,
                    (committed, decision_id),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def list_pending_acknowledgements(self, now: datetime) -> tuple[PendingAcknowledgement, ...]:
        with connect_database(self._database) as connection:
            rows = connection.execute(
                """
                SELECT acknowledgement_id, source_id, source_event_id,
                       ack_operation_id, attempts, next_attempt_at, ttl_hours
                FROM pending_acknowledgements
                WHERE state = 'pending' AND next_attempt_at <= ?
                ORDER BY created_at, acknowledgement_id
                """,
                (_utc_iso(now),),
            ).fetchall()
        return tuple(PendingAcknowledgement(*tuple(row)) for row in rows)

    def mark_acknowledged(self, acknowledgement_id: str, *, acknowledged_at: datetime) -> None:
        timestamp = _utc_iso(acknowledged_at)
        with connect_database(self._database) as connection:
            connection.execute(
                """
                UPDATE pending_acknowledgements
                SET state = 'acknowledged', acknowledged_at = ?, updated_at = ?,
                    last_error = NULL
                WHERE acknowledgement_id = ?
                """,
                (timestamp, timestamp, acknowledgement_id),
            )

    def mark_ack_failed(
        self,
        acknowledgement_id: str,
        *,
        failed_at: datetime,
        error: str,
    ) -> None:
        timestamp = _utc_iso(failed_at)
        with connect_database(self._database) as connection:
            row = connection.execute(
                "SELECT attempts FROM pending_acknowledgements WHERE acknowledgement_id = ?",
                (acknowledgement_id,),
            ).fetchone()
            if row is None:
                raise KeyError(acknowledgement_id)
            attempts = int(row[0]) + 1
            delay = min(3600, 30 * (2 ** (attempts - 1)))
            next_attempt = _utc_iso(failed_at + timedelta(seconds=delay))
            connection.execute(
                """
                UPDATE pending_acknowledgements
                SET attempts = ?, next_attempt_at = ?, last_error = ?, updated_at = ?
                WHERE acknowledgement_id = ?
                """,
                (attempts, next_attempt, error, timestamp, acknowledgement_id),
            )

    def save_context(
        self,
        *,
        source_id: str,
        payload: dict[str, Any],
        fingerprint: str,
        observed_at: datetime,
    ) -> bool:
        timestamp = _utc_iso(observed_at)
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with connect_database(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT fingerprint FROM context_states WHERE source_id = ?",
                    (source_id,),
                ).fetchone()
                changed = row is None or str(row[0]) != fingerprint
                connection.execute(
                    """
                    INSERT INTO context_states(
                        source_id, payload_json, fingerprint, observed_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        fingerprint = excluded.fingerprint,
                        observed_at = excluded.observed_at,
                        updated_at = excluded.updated_at
                    """,
                    (source_id, payload_json, fingerprint, timestamp, timestamp),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return changed

    def load_context(self, source_id: str) -> StoredContext | None:
        with connect_database(self._database) as connection:
            row = connection.execute(
                """
                SELECT source_id, payload_json, fingerprint, observed_at, updated_at
                FROM context_states WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()
        return None if row is None else _stored_context(row)

    def list_contexts(self) -> tuple[StoredContext, ...]:
        with connect_database(self._database) as connection:
            rows = connection.execute(
                """
                SELECT source_id, payload_json, fingerprint, observed_at, updated_at
                FROM context_states ORDER BY source_id
                """
            ).fetchall()
        return tuple(_stored_context(row) for row in rows)

    def claim_context_reevaluation(self, now: datetime, *, min_interval: timedelta) -> bool:
        timestamp = _utc_iso(now)
        with connect_database(self._database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT last_signaled_at FROM context_reevaluation_state WHERE singleton = 1"
                ).fetchone()
                last = _parse_time(row[0]) if row is not None and row[0] else None
                allowed = last is None or now.astimezone(UTC) - last >= min_interval
                if allowed:
                    connection.execute(
                        """
                        INSERT INTO context_reevaluation_state(
                            singleton, last_signaled_at, last_candidate_at,
                            suppressed_count
                        ) VALUES (1, ?, ?, 0)
                        ON CONFLICT(singleton) DO UPDATE SET
                            last_signaled_at = excluded.last_signaled_at,
                            last_candidate_at = excluded.last_candidate_at,
                            suppressed_count = 0
                        """,
                        (timestamp, timestamp),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE context_reevaluation_state
                        SET last_candidate_at = ?,
                            suppressed_count = suppressed_count + 1
                        WHERE singleton = 1
                        """,
                        (timestamp,),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return allowed

    def load_context_reevaluation_state(self) -> dict[str, Any] | None:
        columns = ("last_signaled_at", "last_candidate_at", "suppressed_count")
        with connect_database(self._database) as connection:
            row = connection.execute(
                f"SELECT {', '.join(columns)} FROM context_reevaluation_state WHERE singleton = 1"
            ).fetchone()
        return None if row is None else dict(zip(columns, tuple(row), strict=True))

    def _upsert_state(
        self, table: str, columns: tuple[str, ...], values: tuple[object, ...]
    ) -> None:
        updates = ", ".join(f"{name} = excluded.{name}" for name in columns[1:])
        placeholders = ", ".join("?" for _ in columns)
        with connect_database(self._database) as connection:
            connection.execute(
                f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(session_key) DO UPDATE SET {updates}",
                values,
            )

    def _load_state(
        self, table: str, columns: tuple[str, ...], session_key: str
    ) -> dict[str, Any] | None:
        with connect_database(self._database) as connection:
            row = connection.execute(
                f"SELECT {', '.join(columns)} FROM {table} WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return None if row is None else dict(zip(columns, tuple(row), strict=True))


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("时间必须包含时区")
    return value.astimezone(UTC).isoformat()


def _content_occurrence_expired(row: Sequence[object], fetched_at: datetime) -> bool:
    kind = str(row[2])
    consumed_at = row[6]
    ack_state = None if row[9] is None else str(row[9])
    ttl_hours = None if row[10] is None else int(str(row[10]))
    acknowledged_at = row[11]
    if (
        kind != "content"
        or consumed_at is None
        or ack_state != "acknowledged"
        or ttl_hours is None
        or acknowledged_at is None
    ):
        return False
    eligible_at = _parse_time(acknowledged_at) + timedelta(hours=ttl_hours)
    return fetched_at.astimezone(UTC) >= eligible_at


def _parse_time(value: object) -> datetime:
    return datetime.fromisoformat(str(value)).astimezone(UTC)


def _stored_context(row: Sequence[object]) -> StoredContext:
    return StoredContext(
        source_id=str(row[0]),
        payload=json.loads(str(row[1])),
        fingerprint=str(row[2]),
        observed_at=str(row[3]),
        updated_at=str(row[4]),
    )


def _proactive_decision(row: Sequence[object]) -> ProactiveDecisionRecord:
    payload = json.loads(str(row[8]))
    return ProactiveDecisionRecord(
        decision_id=str(row[0]),
        session_key=str(row[1]),
        trigger_kind=str(row[2]),
        action=str(row[3]),
        source_events=tuple(tuple(item) for item in json.loads(str(row[4]))),
        effect_operation_id=None if row[5] is None else str(row[5]),
        activity_version=int(str(row[6])),
        state=str(row[7]),
        message=str(payload.get("message") or ""),
        evidence=tuple(str(item) for item in payload.get("evidence") or ()),
        reason=str(payload.get("reason") or ""),
        delivery_key=str(payload.get("delivery_key") or ""),
    )


__all__ = [
    "PendingAcknowledgement",
    "StoredProactiveEvent",
    "StoredContext",
    "ProactiveDecisionRecord",
    "ProactiveRepository",
    "stable_proactive_effect_operation_id",
]
