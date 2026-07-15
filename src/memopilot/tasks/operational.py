"""以 operational.db 为事实源的任务仓储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.persistence.migrations import connect_database

Failpoint = Callable[[str], None]


class LostLeaseError(RuntimeError):
    """写入者持有的 fencing epoch 已经过期。"""


class FenceToken(Protocol):
    @property
    def session_key(self) -> str: ...

    @property
    def owner_id(self) -> str: ...

    @property
    def epoch(self) -> int: ...


@dataclass(frozen=True, slots=True)
class InboundCommand:
    event_id: str
    message_id: str
    session_key: str
    channel: str
    chat_id: str
    payload: Mapping[str, Any]
    received_at: datetime


@dataclass(frozen=True, slots=True)
class InterruptCommand:
    event_id: str
    message_id: str
    session_key: str
    channel: str
    chat_id: str
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    job_id: str
    outbox_id: str
    activity_version: int
    created: bool


@dataclass(frozen=True, slots=True)
class InterruptResult:
    activity_version: int
    created: bool
    target_run_id: str | None


@dataclass(frozen=True, slots=True)
class TurnInterruptSnapshot:
    original_message: str
    partial_reply: str = ""
    partial_thinking: str = ""
    tools_used: tuple[str, ...] = ()
    tool_chain: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class TurnInterruptSnapshotRecord(TurnInterruptSnapshot):
    snapshot_id: str = ""
    source_run_id: str = ""
    session_key: str = ""


@dataclass(frozen=True, slots=True)
class SessionIdentityRecord:
    identity_kind: str
    identity_value: str
    session_key: str
    chat_id: str


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    kind: str
    priority: int
    session_key: str
    state: str
    activity_version: int
    payload_json: str


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    outbox_id: str
    event_type: str
    aggregate_id: str
    payload_json: str
    state: str
    attempts: int
    recovery_count: int
    claim_owner: str | None


@dataclass(frozen=True, slots=True)
class RunClaim:
    run_id: str
    job_id: str
    session_key: str
    attempt_no: int
    owner_id: str
    fencing_epoch: int


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_id: str
    run_id: str
    step_index: int
    phase: str
    step_type: str
    state: str
    tool_name: str | None
    input_json: str | None
    observation_json: str | None
    owner_id: str
    fencing_epoch: int


class OperationalRepository:
    """用短事务封装跨进程共享的 operational.db。"""

    _COUNTABLE_TABLES = frozenset(
        {
            "sessions",
            "session_activity",
            "inbound_events",
            "agent_jobs",
            "outbox_events",
            "outbound_effects",
            "runs",
            "run_attempts",
            "session_identities",
            "session_interrupts",
            "turn_interrupt_snapshots",
            "steps",
        }
    )

    def __init__(self, database: Path, *, busy_timeout_seconds: float = 5) -> None:
        self.database = Path(database)
        self.busy_timeout_seconds = busy_timeout_seconds

    def accept_inbound(
        self,
        command: InboundCommand,
        *,
        failpoint: Failpoint | None = None,
    ) -> EnqueueResult:
        """原子写入 inbox、活动版本、P0 Job 与待发布 Outbox。"""
        occurred_at = _utc_iso(command.received_at)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._find_inbound_result(
                connection,
                event_id=command.event_id,
                message_id=command.message_id,
            )
            if existing is not None:
                connection.execute("COMMIT")
                return existing

            connection.execute(
                """
                INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    channel = excluded.channel,
                    chat_id = excluded.chat_id,
                    updated_at = excluded.updated_at
                """,
                (
                    command.session_key,
                    command.channel,
                    command.chat_id,
                    occurred_at,
                    occurred_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO session_activity(
                    session_key, activity_version, last_user_at, updated_at
                ) VALUES (?, 1, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    activity_version = session_activity.activity_version + 1,
                    last_user_at = excluded.last_user_at,
                    updated_at = excluded.updated_at
                """,
                (command.session_key, occurred_at, occurred_at),
            )
            row = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (command.session_key,),
            ).fetchone()
            assert row is not None
            activity_version = int(row["activity_version"])

            job_id = _stable_id("job", f"inbound:{command.message_id}")
            outbox_id = _stable_id("outbox", job_id)
            payload_json = _json(command.payload)
            connection.execute(
                """
                INSERT INTO inbound_events(
                    event_id, message_id, session_key, payload_json, received_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    command.event_id,
                    command.message_id,
                    command.session_key,
                    payload_json,
                    occurred_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO agent_jobs(
                    job_id, kind, priority, session_key, idempotency_key, state,
                    activity_version, payload_json, created_at, updated_at
                ) VALUES (?, 'agent.turn', 0, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    command.session_key,
                    f"inbound:{command.message_id}",
                    activity_version,
                    payload_json,
                    occurred_at,
                    occurred_at,
                ),
            )
            outbox_payload = _json(
                {
                    "activity_version": activity_version,
                    "job_id": job_id,
                    "kind": "agent.turn",
                    "priority": 0,
                    "session_key": command.session_key,
                }
            )
            connection.execute(
                """
                INSERT INTO outbox_events(
                    outbox_id, event_type, aggregate_id, payload_json,
                    idempotency_key, state, next_attempt_at, created_at, updated_at
                ) VALUES (?, 'agent.job.queued', ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    outbox_id,
                    job_id,
                    outbox_payload,
                    f"publish-job:{job_id}",
                    occurred_at,
                    occurred_at,
                    occurred_at,
                ),
            )
            if failpoint is not None:
                failpoint("before_commit")
            connection.execute("COMMIT")
            return EnqueueResult(job_id, outbox_id, activity_version, True)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, kind, priority, session_key, state,
                       activity_version, payload_json
                FROM agent_jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return JobRecord(
            job_id=str(row["job_id"]),
            kind=str(row["kind"]),
            priority=int(row["priority"]),
            session_key=str(row["session_key"]),
            state=str(row["state"]),
            activity_version=int(row["activity_version"]),
            payload_json=str(row["payload_json"]),
        )

    def get_activity_version(self, session_key: str) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return None if row is None else int(row["activity_version"])

    def remember_session_identities(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        identities: Mapping[str, str],
        now: datetime,
    ) -> None:
        allowed_kinds = {"open_id", "user_id", "union_id"}
        normalized = {
            kind: str(value).strip()
            for kind, value in identities.items()
            if kind in allowed_kinds and str(value).strip()
        }
        if not normalized:
            return
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    channel = excluded.channel,
                    chat_id = excluded.chat_id,
                    updated_at = excluded.updated_at
                """,
                (session_key, channel, chat_id, now_text, now_text),
            )
            for kind, value in normalized.items():
                connection.execute(
                    """
                    INSERT INTO session_identities(
                        channel, identity_kind, identity_value, session_key, chat_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(channel, identity_kind, identity_value) DO UPDATE SET
                        session_key = excluded.session_key,
                        chat_id = excluded.chat_id,
                        updated_at = excluded.updated_at
                    """,
                    (channel, kind, value, session_key, chat_id, now_text),
                )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def list_session_identities(self, channel: str) -> tuple[SessionIdentityRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT identity_kind, identity_value, session_key, chat_id
                FROM session_identities
                WHERE channel = ?
                ORDER BY identity_kind, identity_value
                """,
                (channel,),
            ).fetchall()
        return tuple(
            SessionIdentityRecord(
                identity_kind=str(row["identity_kind"]),
                identity_value=str(row["identity_value"]),
                session_key=str(row["session_key"]),
                chat_id=str(row["chat_id"]),
            )
            for row in rows
        )

    def request_interrupt(self, command: InterruptCommand) -> InterruptResult:
        requested_at = _utc_iso(command.requested_at)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                """
                SELECT activity_version, target_run_id FROM session_interrupts
                WHERE event_id = ? OR message_id = ?
                """,
                (command.event_id, command.message_id),
            ).fetchone()
            if existing is not None:
                connection.execute("COMMIT")
                return InterruptResult(
                    int(existing["activity_version"]),
                    False,
                    str(existing["target_run_id"])
                    if existing["target_run_id"] is not None
                    else None,
                )
            connection.execute(
                """
                INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    channel = excluded.channel,
                    chat_id = excluded.chat_id,
                    updated_at = excluded.updated_at
                """,
                (
                    command.session_key,
                    command.channel,
                    command.chat_id,
                    requested_at,
                    requested_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO session_activity(
                    session_key, activity_version, last_user_at, updated_at
                ) VALUES (?, 1, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    activity_version = session_activity.activity_version + 1,
                    last_user_at = excluded.last_user_at,
                    updated_at = excluded.updated_at
                """,
                (command.session_key, requested_at, requested_at),
            )
            row = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (command.session_key,),
            ).fetchone()
            assert row is not None
            activity_version = int(row["activity_version"])
            active = connection.execute(
                """
                SELECT r.run_id
                FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE j.session_key = ? AND r.state = 'running' AND j.state = 'running'
                ORDER BY r.started_at DESC, r.run_id DESC
                LIMIT 1
                """,
                (command.session_key,),
            ).fetchone()
            target_run_id = str(active["run_id"]) if active is not None else None
            connection.execute(
                """
                INSERT INTO session_interrupts(
                    event_id, message_id, session_key, activity_version, requested_at,
                    target_run_id, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command.event_id,
                    command.message_id,
                    command.session_key,
                    activity_version,
                    requested_at,
                    target_run_id,
                    "pending" if target_run_id is not None else "no_active",
                ),
            )
            connection.execute("COMMIT")
            return InterruptResult(activity_version, True, target_run_id)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def has_pending_interrupt(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM session_interrupts
                WHERE target_run_id = ? AND state = 'pending' LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return row is not None

    def finish_interrupted_run(
        self,
        run_id: str,
        *,
        lease: FenceToken,
        snapshot: TurnInterruptSnapshot,
        now: datetime,
        ttl: timedelta = timedelta(minutes=30),
    ) -> str:
        if ttl.total_seconds() <= 0:
            raise ValueError("中断快照 TTL 必须大于 0")
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            run = connection.execute(
                """
                SELECT r.job_id FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE r.run_id = ? AND r.state = 'running' AND j.state = 'running'
                  AND r.owner_id = ? AND r.fencing_epoch = ? AND j.session_key = ?
                """,
                (run_id, lease.owner_id, lease.epoch, lease.session_key),
            ).fetchone()
            pending = connection.execute(
                """
                SELECT 1 FROM session_interrupts
                WHERE target_run_id = ? AND state = 'pending' LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if run is None or pending is None:
                raise LostLeaseError("Run 已失权或没有待处理的定向中断")
            uncertain_effect = connection.execute(
                """
                SELECT 1 FROM outbound_effects
                WHERE run_id = ? AND state IN ('sending', 'unknown', 'needs_review')
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            outcome = "needs_review" if uncertain_effect is not None else "cancelled"
            snapshot_id = _stable_id("interrupt", run_id)
            connection.execute(
                """
                INSERT INTO turn_interrupt_snapshots(
                    snapshot_id, source_run_id, session_key, original_message,
                    partial_reply, partial_thinking, tools_json, tool_chain_json,
                    interrupted_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_run_id) DO NOTHING
                """,
                (
                    snapshot_id,
                    run_id,
                    lease.session_key,
                    snapshot.original_message,
                    snapshot.partial_reply,
                    snapshot.partial_thinking,
                    _json(snapshot.tools_used),
                    _json(snapshot.tool_chain),
                    now_text,
                    _utc_iso(now + ttl),
                ),
            )
            connection.execute(
                """
                UPDATE session_interrupts SET state = 'acknowledged', acknowledged_at = ?
                WHERE target_run_id = ? AND state = 'pending'
                """,
                (now_text, run_id),
            )
            connection.execute(
                """
                UPDATE runs SET state = ?, finished_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (outcome, now_text, now_text, now_text, run_id),
            )
            connection.execute(
                """
                UPDATE run_attempts SET outcome = ?, finished_at = ?, heartbeat_at = ?
                WHERE run_id = ? AND owner_id = ? AND fencing_epoch = ? AND finished_at IS NULL
                """,
                (outcome, now_text, now_text, run_id, lease.owner_id, lease.epoch),
            )
            connection.execute(
                """
                UPDATE agent_jobs
                SET state = ?, heartbeat_at = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (outcome, now_text, now_text, now_text, run["job_id"]),
            )
            connection.execute("COMMIT")
            return snapshot_id
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def reserve_interrupt_snapshot(
        self,
        session_key: str,
        *,
        job_id: str,
        now: datetime,
    ) -> TurnInterruptSnapshotRecord | None:
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT s.* FROM turn_interrupt_snapshots AS s
                JOIN agent_jobs AS j ON j.job_id = ?
                WHERE s.session_key = ? AND s.consumed_at IS NULL AND s.expires_at > ?
                  AND s.interrupted_at < j.created_at
                  AND (s.reserved_job_id IS NULL OR s.reserved_job_id = ?)
                ORDER BY s.interrupted_at DESC LIMIT 1
                """,
                (job_id, session_key, now_text, job_id),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            changed = connection.execute(
                """
                UPDATE turn_interrupt_snapshots SET reserved_job_id = ?
                WHERE snapshot_id = ? AND (reserved_job_id IS NULL OR reserved_job_id = ?)
                """,
                (job_id, row["snapshot_id"], job_id),
            ).rowcount
            if changed != 1:
                connection.execute("ROLLBACK")
                return None
            connection.execute("COMMIT")
            return _interrupt_snapshot_record(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def consume_interrupt_snapshot(
        self, snapshot_id: str, *, job_id: str, now: datetime
    ) -> bool:
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE turn_interrupt_snapshots SET consumed_at = ?
                WHERE snapshot_id = ? AND reserved_job_id = ? AND consumed_at IS NULL
                """,
                (_utc_iso(now), snapshot_id, job_id),
            ).rowcount
        return changed == 1

    def release_interrupt_snapshot(self, snapshot_id: str, *, job_id: str) -> bool:
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE turn_interrupt_snapshots SET reserved_job_id = NULL
                WHERE snapshot_id = ? AND reserved_job_id = ? AND consumed_at IS NULL
                """,
                (snapshot_id, job_id),
            ).rowcount
        return changed == 1

    def claim_next_outbox(
        self,
        *,
        owner_id: str,
        now: datetime,
        claim_ttl: timedelta,
    ) -> OutboxRecord | None:
        now_text = _utc_iso(now)
        claim_until = _utc_iso(now + claim_ttl)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT outbox_id
                FROM outbox_events
                WHERE next_attempt_at <= ?
                  AND (
                    state = 'pending'
                    OR (state = 'publishing' AND claim_until <= ?)
                  )
                ORDER BY created_at, outbox_id
                LIMIT 1
                """,
                (now_text, now_text),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            outbox_id = str(row["outbox_id"])
            changed = connection.execute(
                """
                UPDATE outbox_events
                SET state = 'publishing', attempts = attempts + 1,
                    claim_owner = ?, claim_until = ?, updated_at = ?
                WHERE outbox_id = ?
                  AND (
                    state = 'pending'
                    OR (state = 'publishing' AND claim_until <= ?)
                  )
                """,
                (owner_id, claim_until, now_text, outbox_id, now_text),
            ).rowcount
            if changed != 1:
                connection.execute("ROLLBACK")
                return None
            claimed = connection.execute(
                "SELECT * FROM outbox_events WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
            assert claimed is not None
            connection.execute("COMMIT")
            return _outbox_record(claimed)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


    def mark_outbox_published(
        self,
        outbox_id: str,
        *,
        owner_id: str,
        now: datetime,
    ) -> bool:
        now_text = _utc_iso(now)
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE outbox_events
                SET state = 'published', published_at = ?, claim_owner = NULL,
                    claim_until = NULL, last_error = NULL, updated_at = ?
                WHERE outbox_id = ? AND state = 'publishing' AND claim_owner = ?
                """,
                (now_text, now_text, outbox_id, owner_id),
            ).rowcount
        return changed == 1

    def mark_outbox_retry(
        self,
        outbox_id: str,
        *,
        owner_id: str,
        now: datetime,
        error: str,
    ) -> bool:
        now_text = _utc_iso(now)
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE outbox_events
                SET state = 'pending', claim_owner = NULL, claim_until = NULL,
                    next_attempt_at = ?, last_error = ?, updated_at = ?
                WHERE outbox_id = ? AND state = 'publishing' AND claim_owner = ?
                """,
                (now_text, error[:1000], now_text, outbox_id, owner_id),
            ).rowcount
        return changed == 1

    def get_outbox_for_job(self, job_id: str) -> OutboxRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM outbox_events
                WHERE aggregate_id = ? AND event_type = 'agent.job.queued'
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _outbox_record(row)

    def queued_job_ids(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id FROM agent_jobs WHERE state = 'queued' ORDER BY created_at, job_id"
            ).fetchall()
        return tuple(str(row["job_id"]) for row in rows)

    def requeue_published_outboxes(
        self, job_ids: tuple[str, ...], *, now: datetime
    ) -> tuple[str, ...]:
        if not job_ids:
            return ()
        now_text = _utc_iso(now)
        placeholders = ",".join("?" for _ in job_ids)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            rows = connection.execute(
                f"""
                SELECT aggregate_id FROM outbox_events
                WHERE event_type = 'agent.job.queued'
                  AND state = 'published'
                  AND aggregate_id IN ({placeholders})
                ORDER BY created_at, aggregate_id
                """,
                job_ids,
            ).fetchall()
            recovered = tuple(str(row["aggregate_id"]) for row in rows)
            if not recovered:
                connection.execute("COMMIT")
                return ()
            recovered_placeholders = ",".join("?" for _ in recovered)
            connection.execute(
                f"""
                UPDATE outbox_events
                SET state = 'pending', claim_owner = NULL, claim_until = NULL,
                    published_at = NULL, next_attempt_at = ?,
                    recovery_count = recovery_count + 1, updated_at = ?
                WHERE event_type = 'agent.job.queued'
                  AND state = 'published'
                  AND aggregate_id IN ({recovered_placeholders})
                """,
                (now_text, now_text, *recovered),
            )
            connection.execute("COMMIT")
            return recovered
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def allocate_fence(self, session_key: str, *, owner_id: str, now: datetime) -> int:
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            session = connection.execute(
                "SELECT 1 FROM sessions WHERE session_key = ?", (session_key,)
            ).fetchone()
            if session is None:
                raise KeyError(f"会话不存在: {session_key}")
            connection.execute(
                """
                INSERT INTO session_fences(
                    session_key, current_epoch, owner_id, heartbeat_at, updated_at
                ) VALUES (?, 0, NULL, NULL, ?)
                ON CONFLICT(session_key) DO NOTHING
                """,
                (session_key, now_text),
            )
            connection.execute(
                """
                UPDATE session_fences
                SET current_epoch = current_epoch + 1,
                    owner_id = ?, heartbeat_at = ?, updated_at = ?
                WHERE session_key = ?
                """,
                (owner_id, now_text, now_text, session_key),
            )
            row = connection.execute(
                "SELECT current_epoch FROM session_fences WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            assert row is not None
            connection.execute("COMMIT")
            return int(row["current_epoch"])
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def heartbeat_fence(self, lease: FenceToken, *, now: datetime) -> bool:
        now_text = _utc_iso(now)
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE session_fences
                SET heartbeat_at = ?, updated_at = ?
                WHERE session_key = ? AND owner_id = ? AND current_epoch = ?
                """,
                (now_text, now_text, lease.session_key, lease.owner_id, lease.epoch),
            ).rowcount
        return changed == 1

    def claim_job(
        self,
        job_id: str,
        *,
        lease: FenceToken,
        now: datetime,
    ) -> RunClaim | None:
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            job = connection.execute(
                "SELECT state, session_key FROM agent_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(job_id)
            if str(job["session_key"]) != lease.session_key:
                raise ValueError("Job 与 lease 不属于同一会话")
            existing = connection.execute(
                "SELECT * FROM runs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if str(job["state"]) == "running":
                if (
                    existing is not None
                    and existing["owner_id"] == lease.owner_id
                    and existing["fencing_epoch"] == lease.epoch
                ):
                    attempt = connection.execute(
                        "SELECT MAX(attempt_no) FROM run_attempts WHERE run_id = ?",
                        (existing["run_id"],),
                    ).fetchone()
                    assert attempt is not None and attempt[0] is not None
                    connection.execute("COMMIT")
                    return RunClaim(
                        str(existing["run_id"]),
                        job_id,
                        lease.session_key,
                        int(attempt[0]),
                        lease.owner_id,
                        lease.epoch,
                    )
                connection.execute("COMMIT")
                return None
            if str(job["state"]) != "queued":
                connection.execute("COMMIT")
                return None

            connection.execute(
                """
                UPDATE agent_jobs
                SET state = 'running', heartbeat_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'queued'
                """,
                (now_text, now_text, job_id),
            )
            if existing is None:
                run_id = _stable_id("run", job_id)
                attempt_no = 1
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, job_id, owner_id, fencing_epoch, state,
                        started_at, heartbeat_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        job_id,
                        lease.owner_id,
                        lease.epoch,
                        now_text,
                        now_text,
                        now_text,
                        now_text,
                    ),
                )
            else:
                if str(existing["state"]) != "recovering":
                    raise RuntimeError("已有 Run 不处于可恢复状态")
                run_id = str(existing["run_id"])
                attempt_row = connection.execute(
                    "SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM run_attempts WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                assert attempt_row is not None
                attempt_no = int(attempt_row[0])
                connection.execute(
                    """
                    UPDATE runs
                    SET owner_id = ?, fencing_epoch = ?, state = 'running',
                        heartbeat_at = ?, updated_at = ?
                    WHERE run_id = ? AND state = 'recovering'
                    """,
                    (lease.owner_id, lease.epoch, now_text, now_text, run_id),
                )
            attempt_id = _stable_id("attempt", f"{run_id}:{attempt_no}")
            connection.execute(
                """
                INSERT INTO run_attempts(
                    attempt_id, run_id, attempt_no, owner_id, fencing_epoch,
                    started_at, heartbeat_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    run_id,
                    attempt_no,
                    lease.owner_id,
                    lease.epoch,
                    now_text,
                    now_text,
                ),
            )
            connection.execute("COMMIT")
            return RunClaim(
                run_id,
                job_id,
                lease.session_key,
                attempt_no,
                lease.owner_id,
                lease.epoch,
            )
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def heartbeat_run(self, run_id: str, *, lease: FenceToken, now: datetime) -> None:
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            run = connection.execute(
                """
                SELECT r.job_id FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE r.run_id = ? AND r.state = 'running'
                  AND r.owner_id = ? AND r.fencing_epoch = ?
                  AND j.session_key = ?
                """,
                (run_id, lease.owner_id, lease.epoch, lease.session_key),
            ).fetchone()
            if run is None:
                raise LostLeaseError("Run 已不属于当前 lease")
            connection.execute(
                "UPDATE runs SET heartbeat_at = ?, updated_at = ? WHERE run_id = ?",
                (now_text, now_text, run_id),
            )
            connection.execute(
                "UPDATE agent_jobs SET heartbeat_at = ?, updated_at = ? WHERE job_id = ?",
                (now_text, now_text, run["job_id"]),
            )
            connection.execute(
                """
                UPDATE run_attempts SET heartbeat_at = ?
                WHERE run_id = ? AND owner_id = ? AND fencing_epoch = ? AND finished_at IS NULL
                """,
                (now_text, run_id, lease.owner_id, lease.epoch),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def authorize_live_progress(
        self,
        run_id: str,
        *,
        lease: FenceToken,
        expected_activity_version: int,
        now: datetime,
        creating: bool,
        cancel_on_activity: bool = True,
    ) -> bool:
        """原子确认 live 副作用仍属于当前 Run，且创建未超过一小时窗口。"""
        creation_cutoff = _utc_iso(now - timedelta(hours=1))
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                JOIN session_activity AS a ON a.session_key = j.session_key
                JOIN session_fences AS f ON f.session_key = j.session_key
                WHERE r.run_id = ?
                  AND r.state = 'running' AND j.state = 'running'
                  AND r.owner_id = ? AND r.fencing_epoch = ?
                  AND j.session_key = ?
                  AND f.owner_id = ? AND f.current_epoch = ?
                  AND j.activity_version = ?
                  AND (? = 0 OR a.activity_version = ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM session_interrupts AS i
                      WHERE i.target_run_id = r.run_id AND i.state = 'pending'
                  )
                  AND (? = 0 OR r.started_at >= ?)
                """,
                (
                    run_id,
                    lease.owner_id,
                    lease.epoch,
                    lease.session_key,
                    lease.owner_id,
                    lease.epoch,
                    expected_activity_version,
                    int(cancel_on_activity),
                    expected_activity_version,
                    int(creating),
                    creation_cutoff,
                ),
            ).fetchone()
        return row is not None

    def append_step(
        self,
        run_id: str,
        *,
        lease: FenceToken,
        phase: str,
        step_type: str,
        state: str,
        now: datetime,
        tool_name: str | None = None,
        input: Mapping[str, Any] | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> StepRecord:
        """以当前 fencing 身份向 Run 追加一个不可变执行步骤。"""
        allowed_states = {
            "pending",
            "running",
            "succeeded",
            "failed",
            "skipped",
            "cancelled",
            "unknown",
        }
        if state not in allowed_states:
            raise ValueError(f"不支持的 Step 状态: {state}")
        if not phase or not step_type:
            raise ValueError("Step phase 和 step_type 不能为空")
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            run = connection.execute(
                """
                SELECT 1 FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE r.run_id = ? AND r.state = 'running'
                  AND r.owner_id = ? AND r.fencing_epoch = ?
                  AND j.session_key = ?
                """,
                (run_id, lease.owner_id, lease.epoch, lease.session_key),
            ).fetchone()
            if run is None:
                raise LostLeaseError("Run 已不属于当前 lease")
            index_row = connection.execute(
                "SELECT COALESCE(MAX(step_index), -1) + 1 FROM steps WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            assert index_row is not None
            step_index = int(index_row[0])
            step_id = _stable_id("step", f"{run_id}:{step_index}")
            input_json = (
                json.dumps(input, ensure_ascii=False, sort_keys=True, default=str)
                if input is not None
                else None
            )
            observation_json = (
                json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str)
                if observation is not None
                else None
            )
            connection.execute(
                """
                INSERT INTO steps(
                    step_id, run_id, step_index, phase, step_type, state,
                    tool_name, input_json, observation_json, owner_id,
                    fencing_epoch, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    run_id,
                    step_index,
                    phase,
                    step_type,
                    state,
                    tool_name,
                    input_json,
                    observation_json,
                    lease.owner_id,
                    lease.epoch,
                    now_text,
                    now_text,
                ),
            )
            connection.execute("COMMIT")
            return StepRecord(
                step_id=step_id,
                run_id=run_id,
                step_index=step_index,
                phase=phase,
                step_type=step_type,
                state=state,
                tool_name=tool_name,
                input_json=input_json,
                observation_json=observation_json,
                owner_id=lease.owner_id,
                fencing_epoch=lease.epoch,
            )
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def list_steps(self, run_id: str) -> tuple[StepRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index",
                (run_id,),
            ).fetchall()
        return tuple(
            StepRecord(
                step_id=str(row["step_id"]),
                run_id=str(row["run_id"]),
                step_index=int(row["step_index"]),
                phase=str(row["phase"]),
                step_type=str(row["step_type"]),
                state=str(row["state"]),
                tool_name=str(row["tool_name"]) if row["tool_name"] is not None else None,
                input_json=str(row["input_json"]) if row["input_json"] is not None else None,
                observation_json=(
                    str(row["observation_json"])
                    if row["observation_json"] is not None
                    else None
                ),
                owner_id=str(row["owner_id"]),
                fencing_epoch=int(row["fencing_epoch"]),
            )
            for row in rows
        )

    def recover_stale_job(
        self,
        job_id: str,
        *,
        lease: FenceToken,
        now: datetime,
        heartbeat_before: datetime,
    ) -> str | None:
        now_text = _utc_iso(now)
        cutoff = _utc_iso(heartbeat_before)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            run = connection.execute(
                """
                SELECT r.run_id, r.state, COALESCE(r.heartbeat_at, r.updated_at) AS last_seen
                FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE r.job_id = ? AND j.session_key = ?
                  AND j.state = 'running' AND r.state = 'running'
                """,
                (job_id, lease.session_key),
            ).fetchone()
            if run is None or str(run["last_seen"]) > cutoff:
                connection.execute("COMMIT")
                return None
            uncertain = connection.execute(
                """
                SELECT 1 FROM outbound_effects
                WHERE run_id = ? AND state IN ('sending', 'unknown', 'needs_review')
                LIMIT 1
                """,
                (run["run_id"],),
            ).fetchone()
            if uncertain is not None:
                target = "needs_review"
                attempt_outcome = "needs_review"
            else:
                target = "recovering"
                attempt_outcome = "lost_lease"
            connection.execute(
                "UPDATE agent_jobs SET state = ?, heartbeat_at = NULL, updated_at = ? "
                "WHERE job_id = ? AND state = 'running'",
                ("needs_review" if uncertain else "queued", now_text, job_id),
            )
            connection.execute(
                """
                UPDATE runs
                SET state = ?, owner_id = NULL, fencing_epoch = NULL,
                    heartbeat_at = NULL, finished_at = ?, updated_at = ?
                WHERE run_id = ? AND state = 'running'
                """,
                (
                    target,
                    now_text if uncertain else None,
                    now_text,
                    run["run_id"],
                ),
            )
            connection.execute(
                """
                UPDATE run_attempts
                SET outcome = ?, finished_at = ?
                WHERE run_id = ? AND finished_at IS NULL
                """,
                (attempt_outcome, now_text, run["run_id"]),
            )
            connection.execute("COMMIT")
            return target
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def is_job_reclaimable(self, job_id: str, *, heartbeat_before: datetime) -> bool:
        return (
            self.pending_recovery_disposition(
                job_id,
                heartbeat_before=heartbeat_before,
            )
            == "resume"
        )

    def pending_recovery_disposition(
        self,
        job_id: str,
        *,
        heartbeat_before: datetime,
    ) -> Literal["resume", "cleanup"] | None:
        cutoff = _utc_iso(heartbeat_before)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT j.state, COALESCE(r.heartbeat_at, r.updated_at) AS last_seen
                FROM agent_jobs AS j
                LEFT JOIN runs AS r ON r.job_id = j.job_id
                WHERE j.job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        state = str(row["state"])
        if state == "queued":
            return "resume"
        if state in {"succeeded", "skipped", "failed", "cancelled", "needs_review"}:
            return "cleanup"
        reclaimable = (
            state == "running" and row["last_seen"] is not None and str(row["last_seen"]) <= cutoff
        )
        return "resume" if reclaimable else None

    def finish_job(
        self,
        run_id: str,
        *,
        lease: FenceToken,
        outcome: str,
        now: datetime,
    ) -> None:
        if outcome not in {"succeeded", "failed", "cancelled", "needs_review"}:
            raise ValueError(f"不支持的 Run 终态: {outcome}")
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            run = connection.execute(
                """
                SELECT r.job_id FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE r.run_id = ? AND r.state = 'running'
                  AND r.owner_id = ? AND r.fencing_epoch = ?
                  AND j.session_key = ?
                """,
                (run_id, lease.owner_id, lease.epoch, lease.session_key),
            ).fetchone()
            if run is None:
                raise LostLeaseError("Run 已不属于当前 lease")
            connection.execute(
                """
                UPDATE runs
                SET state = ?, finished_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (outcome, now_text, now_text, now_text, run_id),
            )
            connection.execute(
                """
                UPDATE run_attempts
                SET outcome = ?, finished_at = ?, heartbeat_at = ?
                WHERE run_id = ? AND owner_id = ? AND fencing_epoch = ?
                  AND finished_at IS NULL
                """,
                (outcome, now_text, now_text, run_id, lease.owner_id, lease.epoch),
            )
            connection.execute(
                """
                UPDATE agent_jobs
                SET state = ?, heartbeat_at = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (outcome, now_text, now_text, now_text, run["job_id"]),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def resolve_needs_review(
        self,
        run_id: str,
        *,
        lease: FenceToken,
        outcome: str,
        now: datetime,
    ) -> None:
        """在显式核对外部副作用后关闭 needs_review Run。"""
        if outcome not in {"succeeded", "failed", "cancelled"}:
            raise ValueError(f"不支持的核对终态: {outcome}")
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            existing = connection.execute(
                """
                SELECT r.job_id, r.state AS run_state, j.state AS job_state
                FROM runs AS r JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE r.run_id = ? AND j.session_key = ?
                """,
                (run_id, lease.session_key),
            ).fetchone()
            if existing is None:
                raise LostLeaseError("待核对 Run 已不存在或不属于当前会话")
            if existing["run_state"] == outcome and existing["job_state"] == outcome:
                connection.execute("COMMIT")
                return
            run = connection.execute(
                """
                SELECT r.job_id FROM runs AS r
                JOIN agent_jobs AS j ON j.job_id = r.job_id
                WHERE r.run_id = ? AND r.state = 'needs_review'
                  AND j.state = 'needs_review' AND j.session_key = ?
                """,
                (run_id, lease.session_key),
            ).fetchone()
            if run is None:
                raise LostLeaseError("待核对 Run 已不处于 needs_review")
            connection.execute(
                """
                UPDATE runs
                SET state = ?, owner_id = ?, fencing_epoch = ?,
                    finished_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE run_id = ? AND state = 'needs_review'
                """,
                (
                    outcome,
                    lease.owner_id,
                    lease.epoch,
                    now_text,
                    now_text,
                    now_text,
                    run_id,
                ),
            )
            connection.execute(
                """
                UPDATE agent_jobs
                SET state = ?, heartbeat_at = ?, finished_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'needs_review'
                """,
                (outcome, now_text, now_text, now_text, run["job_id"]),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _require_current_fence(connection: sqlite3.Connection, lease: FenceToken) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM session_fences
            WHERE session_key = ? AND owner_id = ? AND current_epoch = ?
            """,
            (lease.session_key, lease.owner_id, lease.epoch),
        ).fetchone()
        if row is None:
            raise LostLeaseError(
                f"会话 {lease.session_key} 的 owner/epoch 已失效: {lease.owner_id}/{lease.epoch}"
            )

    def count(self, table: str) -> int:
        if table not in self._COUNTABLE_TABLES:
            raise ValueError(f"不允许统计表: {table}")
        with self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        assert row is not None
        return int(row[0])

    def _connect(self) -> sqlite3.Connection:
        return connect_database(
            self.database,
            busy_timeout_seconds=self.busy_timeout_seconds,
        )

    @staticmethod
    def _find_inbound_result(
        connection: sqlite3.Connection,
        *,
        event_id: str,
        message_id: str,
    ) -> EnqueueResult | None:
        row = connection.execute(
            """
            SELECT aj.job_id, aj.activity_version, oe.outbox_id
            FROM inbound_events AS incoming
            JOIN agent_jobs AS aj
              ON aj.idempotency_key = 'inbound:' || incoming.message_id
            JOIN outbox_events AS oe
              ON oe.aggregate_id = aj.job_id
             AND oe.event_type = 'agent.job.queued'
            WHERE incoming.event_id = ? OR incoming.message_id = ?
            LIMIT 1
            """,
            (event_id, message_id),
        ).fetchone()
        if row is None:
            return None
        return EnqueueResult(
            job_id=str(row["job_id"]),
            outbox_id=str(row["outbox_id"]),
            activity_version=int(row["activity_version"]),
            created=False,
        )


def _stable_id(kind: str, identity: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"memopilot:{kind}:{identity}"))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("持久化时间必须包含时区")
    return value.astimezone(UTC).isoformat()


def _outbox_record(row: sqlite3.Row) -> OutboxRecord:
    return OutboxRecord(
        outbox_id=str(row["outbox_id"]),
        event_type=str(row["event_type"]),
        aggregate_id=str(row["aggregate_id"]),
        payload_json=str(row["payload_json"]),
        state=str(row["state"]),
        attempts=int(row["attempts"]),
        recovery_count=int(row["recovery_count"]),
        claim_owner=None if row["claim_owner"] is None else str(row["claim_owner"]),
    )


def _interrupt_snapshot_record(row: sqlite3.Row) -> TurnInterruptSnapshotRecord:
    return TurnInterruptSnapshotRecord(
        snapshot_id=str(row["snapshot_id"]),
        source_run_id=str(row["source_run_id"]),
        session_key=str(row["session_key"]),
        original_message=str(row["original_message"]),
        partial_reply=str(row["partial_reply"]),
        partial_thinking=str(row["partial_thinking"]),
        tools_used=tuple(str(item) for item in json.loads(row["tools_json"])),
        tool_chain=tuple(json.loads(row["tool_chain_json"])),
    )


__all__ = [
    "EnqueueResult",
    "FenceToken",
    "InboundCommand",
    "InterruptCommand",
    "InterruptResult",
    "JobRecord",
    "LostLeaseError",
    "OperationalRepository",
    "OutboxRecord",
    "RunClaim",
    "SessionIdentityRecord",
    "StepRecord",
    "TurnInterruptSnapshot",
    "TurnInterruptSnapshotRecord",
]
