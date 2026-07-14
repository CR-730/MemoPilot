"""以 operational.db 为事实源的 outbound effect 状态机。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from memopilot.persistence.migrations import connect_database
from memopilot.tasks.operational import FenceToken, LostLeaseError


class EffectTransition(StrEnum):
    SEND = "send"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class EffectRequest:
    operation_id: str
    run_id: str
    session_key: str
    channel: str
    chat_id: str
    text: str
    expected_activity_version: int
    lease: FenceToken
    now: datetime


@dataclass(frozen=True, slots=True)
class EffectRecord:
    operation_id: str
    run_id: str
    session_key: str
    channel: str
    chat_id: str
    payload_json: str
    payload_hash: str
    provider_uuid: str
    expected_activity_version: int
    state: str
    owner_id: str
    fencing_epoch: int
    message_id: str | None
    first_requested_at: str | None

    @property
    def text(self) -> str:
        return str(json.loads(self.payload_json)["text"])


class EffectRepository:
    def __init__(self, database: Path, *, busy_timeout_seconds: float = 5) -> None:
        self.database = Path(database)
        self.busy_timeout_seconds = busy_timeout_seconds

    def create(self, request: EffectRequest) -> EffectRecord:
        payload_json = _payload_json(request.channel, request.chat_id, request.text)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        provider_uuid = str(uuid5(NAMESPACE_URL, f"feishu:{request.operation_id}"))
        now = _utc_iso(request.now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_run(connection, request.run_id, request.lease)
            existing = connection.execute(
                "SELECT * FROM outbound_effects WHERE operation_id = ?",
                (request.operation_id,),
            ).fetchone()
            if existing is not None:
                record = _record(existing)
                expected = (
                    request.run_id,
                    request.session_key,
                    request.channel,
                    request.chat_id,
                    payload_hash,
                    provider_uuid,
                    request.expected_activity_version,
                )
                actual = (
                    record.run_id,
                    record.session_key,
                    record.channel,
                    record.chat_id,
                    record.payload_hash,
                    record.provider_uuid,
                    record.expected_activity_version,
                )
                if actual != expected:
                    raise ValueError("同一 operation_id 不允许改变 payload 或发送身份")
                connection.execute("COMMIT")
                return record
            connection.execute(
                """
                INSERT INTO outbound_effects(
                    operation_id, run_id, session_key, payload_hash, provider_uuid,
                    expected_activity_version, state, owner_id, fencing_epoch,
                    created_at, updated_at, channel, chat_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.operation_id,
                    request.run_id,
                    request.session_key,
                    payload_hash,
                    provider_uuid,
                    request.expected_activity_version,
                    request.lease.owner_id,
                    request.lease.epoch,
                    now,
                    now,
                    request.channel,
                    request.chat_id,
                    payload_json,
                ),
            )
            row = connection.execute(
                "SELECT * FROM outbound_effects WHERE operation_id = ?",
                (request.operation_id,),
            ).fetchone()
            assert row is not None
            connection.execute("COMMIT")
            return _record(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def begin_send(
        self,
        operation_id: str,
        *,
        lease: FenceToken,
        now: datetime,
    ) -> EffectTransition:
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            row = connection.execute(
                "SELECT * FROM outbound_effects WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(operation_id)
            record = _record(row)
            self._require_current_run(connection, record.run_id, lease)
            if record.state == "confirmed":
                connection.execute("COMMIT")
                return EffectTransition.CONFIRMED
            if record.state == "cancelled":
                connection.execute("COMMIT")
                return EffectTransition.CANCELLED
            if record.state in {"sending", "needs_review"}:
                connection.execute(
                    "UPDATE outbound_effects SET state = 'needs_review', updated_at = ? "
                    "WHERE operation_id = ?",
                    (now_text, operation_id),
                )
                connection.execute("COMMIT")
                return EffectTransition.NEEDS_REVIEW
            if record.state == "unknown":
                connection.execute("COMMIT")
                return EffectTransition.NEEDS_REVIEW
            activity = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (record.session_key,),
            ).fetchone()
            if (
                activity is None
                or int(activity["activity_version"]) != record.expected_activity_version
            ):
                connection.execute(
                    "UPDATE outbound_effects SET state = 'cancelled', updated_at = ? "
                    "WHERE operation_id = ?",
                    (now_text, operation_id),
                )
                connection.execute("COMMIT")
                return EffectTransition.CANCELLED
            connection.execute(
                """
                UPDATE outbound_effects
                SET state = 'sending', owner_id = ?, fencing_epoch = ?,
                    first_requested_at = COALESCE(first_requested_at, ?),
                    last_attempt_at = ?, updated_at = ?
                WHERE operation_id = ? AND state IN ('pending', 'unknown')
                """,
                (lease.owner_id, lease.epoch, now_text, now_text, now_text, operation_id),
            )
            connection.execute("COMMIT")
            return EffectTransition.SEND
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def mark_confirmed(
        self,
        operation_id: str,
        *,
        lease: FenceToken,
        message_id: str,
        now: datetime,
    ) -> None:
        self._finish_sending(
            operation_id,
            lease=lease,
            state="confirmed",
            now=now,
            message_id=message_id,
        )

    def begin_reconciliation(
        self,
        operation_id: str,
        *,
        lease: FenceToken,
        now: datetime,
    ) -> EffectTransition:
        """在人工确认远端未成功后，显式取得一次安全重试权。"""
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            row = connection.execute(
                "SELECT * FROM outbound_effects WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(operation_id)
            record = _record(row)
            if record.session_key != lease.session_key:
                raise LostLeaseError("核对重试的会话与 lease 不匹配")
            if record.state == "confirmed":
                connection.execute("COMMIT")
                return EffectTransition.CONFIRMED
            if record.state == "cancelled":
                connection.execute("COMMIT")
                return EffectTransition.CANCELLED
            if record.state != "unknown" or record.first_requested_at is None:
                connection.execute("COMMIT")
                return EffectTransition.NEEDS_REVIEW
            first_requested = datetime.fromisoformat(record.first_requested_at)
            if now > first_requested + timedelta(hours=1):
                connection.execute(
                    "UPDATE outbound_effects SET state = 'needs_review', updated_at = ? "
                    "WHERE operation_id = ?",
                    (now_text, operation_id),
                )
                connection.execute("COMMIT")
                return EffectTransition.NEEDS_REVIEW
            activity = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (record.session_key,),
            ).fetchone()
            if (
                activity is None
                or int(activity["activity_version"]) != record.expected_activity_version
            ):
                connection.execute(
                    "UPDATE outbound_effects SET state = 'cancelled', updated_at = ? "
                    "WHERE operation_id = ?",
                    (now_text, operation_id),
                )
                connection.execute("COMMIT")
                return EffectTransition.CANCELLED
            changed = connection.execute(
                """
                UPDATE outbound_effects
                SET state = 'sending', owner_id = ?, fencing_epoch = ?,
                    last_attempt_at = ?, updated_at = ?
                WHERE operation_id = ? AND state = 'unknown'
                """,
                (lease.owner_id, lease.epoch, now_text, now_text, operation_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("outbound effect 核对重试状态发生并发变化")
            connection.execute("COMMIT")
            return EffectTransition.SEND
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def mark_unknown(
        self,
        operation_id: str,
        *,
        lease: FenceToken,
        error: str,
        now: datetime,
    ) -> None:
        self._finish_sending(
            operation_id,
            lease=lease,
            state="unknown",
            now=now,
            error=error,
        )

    def mark_known_failure(
        self,
        operation_id: str,
        *,
        lease: FenceToken,
        error: str,
        now: datetime,
    ) -> None:
        self._finish_sending(
            operation_id,
            lease=lease,
            state="needs_review",
            now=now,
            error=error,
        )

    def get(self, operation_id: str) -> EffectRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM outbound_effects WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return None if row is None else _record(row)

    def for_run(self, run_id: str) -> EffectRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM outbound_effects WHERE run_id = ? ORDER BY created_at LIMIT 1",
                (run_id,),
            ).fetchone()
        return None if row is None else _record(row)

    def _finish_sending(
        self,
        operation_id: str,
        *,
        lease: FenceToken,
        state: str,
        now: datetime,
        message_id: str | None = None,
        error: str | None = None,
    ) -> None:
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._require_current_fence(connection, lease)
            changed = connection.execute(
                """
                UPDATE outbound_effects
                SET state = ?, message_id = COALESCE(?, message_id),
                    confirmed_at = CASE WHEN ? = 'confirmed' THEN ? ELSE confirmed_at END,
                    error_json = ?, updated_at = ?
                WHERE operation_id = ? AND state = 'sending'
                  AND owner_id = ? AND fencing_epoch = ?
                """,
                (
                    state,
                    message_id,
                    state,
                    now_text,
                    json.dumps({"message": error}, ensure_ascii=False) if error else None,
                    now_text,
                    operation_id,
                    lease.owner_id,
                    lease.epoch,
                ),
            ).rowcount
            if changed != 1:
                raise LostLeaseError("outbound effect 已不属于当前发送者")
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
            raise LostLeaseError("outbound effect fencing 身份已失效")

    @classmethod
    def _require_current_run(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
        lease: FenceToken,
    ) -> None:
        cls._require_current_fence(connection, lease)
        row = connection.execute(
            """
            SELECT 1 FROM runs AS r
            JOIN agent_jobs AS j ON j.job_id = r.job_id
            WHERE r.run_id = ? AND r.state = 'running'
              AND r.owner_id = ? AND r.fencing_epoch = ?
              AND j.session_key = ?
            """,
            (run_id, lease.owner_id, lease.epoch, lease.session_key),
        ).fetchone()
        if row is None:
            raise LostLeaseError("outbound effect 对应 Run 已不属于当前发送者")

    def _connect(self) -> sqlite3.Connection:
        return connect_database(
            self.database,
            busy_timeout_seconds=self.busy_timeout_seconds,
        )


def _payload_json(channel: str, chat_id: str, text: str) -> str:
    return json.dumps(
        {"channel": channel, "chat_id": chat_id, "text": text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _record(row: sqlite3.Row) -> EffectRecord:
    return EffectRecord(
        operation_id=str(row["operation_id"]),
        run_id=str(row["run_id"]),
        session_key=str(row["session_key"]),
        channel=str(row["channel"]),
        chat_id=str(row["chat_id"]),
        payload_json=str(row["payload_json"]),
        payload_hash=str(row["payload_hash"]),
        provider_uuid=str(row["provider_uuid"]),
        expected_activity_version=int(row["expected_activity_version"]),
        state=str(row["state"]),
        owner_id=str(row["owner_id"]),
        fencing_epoch=int(row["fencing_epoch"]),
        message_id=str(row["message_id"]) if row["message_id"] is not None else None,
        first_requested_at=(
            str(row["first_requested_at"])
            if row["first_requested_at"] is not None
            else None
        ),
    )


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("持久化时间必须包含时区")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


__all__ = [
    "EffectRecord",
    "EffectRepository",
    "EffectRequest",
    "EffectTransition",
]
