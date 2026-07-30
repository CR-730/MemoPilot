"""基于 operational.db 的定时任务事实仓储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from memopilot.persistence.migrations import connect_database
from memopilot.scheduling.contracts import (
    CreateSchedule,
    DueScanResult,
    ExecutionMode,
    ScheduledExecution,
    ScheduledTask,
    ScheduleKind,
)
from memopilot.scheduling.time_rules import advance_every, is_cron_expr, parse_duration
from memopilot.tasks.agent_task import AgentTask


class ScheduleRepository:
    """以短事务创建、查询、取消并投递到期定时任务。"""

    def __init__(
        self,
        database: Path,
        *,
        busy_timeout_seconds: float = 5,
    ) -> None:
        self.database = Path(database)
        self.busy_timeout_seconds = busy_timeout_seconds

    def create(self, command: CreateSchedule) -> ScheduledTask:
        created_at = _as_utc(command.created_at)
        next_run_at = _as_utc(command.next_run_at)
        identity = _json(
            {
                "created_at": created_at.isoformat(),
                "expression": command.schedule_expression,
                "mode": command.execution_mode,
                "name": command.name,
                "payload": command.payload,
                "session_key": command.session_key,
                "type": command.schedule_kind,
                "timezone": command.timezone,
            }
        )
        task_id = _stable_id("schedule", identity)
        schedule_json = _json(
            {
                "expression": command.schedule_expression,
                "name": command.name,
                "timezone": command.timezone,
            }
        )
        payload_json = _json(command.payload)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO scheduled_tasks(
                    task_id, session_key, schedule_kind, schedule_json,
                    execution_mode, payload_json, next_run_at, enabled,
                    version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                """,
                (
                    task_id,
                    command.session_key,
                    command.schedule_kind,
                    schedule_json,
                    command.execution_mode,
                    payload_json,
                    next_run_at.isoformat(),
                    created_at.isoformat(),
                    created_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT session_key, schedule_kind, schedule_json, execution_mode,
                       payload_json, created_at
                FROM scheduled_tasks WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            expected = (
                command.session_key,
                command.schedule_kind,
                schedule_json,
                command.execution_mode,
                payload_json,
                created_at.isoformat(),
            )
            if row is None or tuple(row) != expected:
                raise ValueError("task_id 已被不同参数占用")
        result = self.get(task_id)
        assert result is not None
        return result

    def get(self, task_id: str) -> ScheduledTask | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scheduled_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return None if row is None else _task_from_row(row)

    def list_for_session(self, session_key: str) -> tuple[ScheduledTask, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM scheduled_tasks
                WHERE session_key = ? AND enabled = 1
                ORDER BY next_run_at, created_at, task_id
                """,
                (session_key,),
            ).fetchall()
        return tuple(_task_from_row(row) for row in rows)

    def cancel(
        self,
        session_key: str,
        *,
        task_id: str | None = None,
        name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[str, ...]:
        if not task_id and not name:
            raise ValueError("必须提供 task_id 或 name")
        updated_at = _as_utc(now or datetime.now(UTC)).isoformat()
        condition = (
            "task.task_id = ?" if task_id else "json_extract(task.schedule_json, '$.name') = ?"
        )
        value = task_id or name
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pending_clause = (
                    "EXISTS (SELECT 1 FROM scheduled_executions AS execution "
                    "WHERE execution.task_id = task.task_id "
                    "AND execution.state IN ('queued', 'running'))"
                )
                rows = connection.execute(
                    f"SELECT task_id FROM scheduled_tasks AS task "  # noqa: S608
                    f"WHERE session_key = ? AND {condition} "
                    f"AND (enabled = 1 OR {pending_clause})",
                    (session_key, value),
                ).fetchall()
                identifiers = tuple(str(row["task_id"]) for row in rows)
                if identifiers:
                    placeholders = ",".join("?" for _ in identifiers)
                    connection.execute(
                        f"UPDATE scheduled_tasks SET enabled = 0, next_run_at = NULL, "  # noqa: S608
                        f"version = version + 1, updated_at = ? "
                        f"WHERE task_id IN ({placeholders})",
                        (updated_at, *identifiers),
                    )
                    connection.execute(
                        f"UPDATE scheduled_executions SET state = 'cancelled', "  # noqa: S608
                        f"updated_at = ? WHERE task_id IN ({placeholders}) "
                        "AND state IN (?, ?)",
                        (updated_at, *identifiers, "queued", "running"),
                    )
                connection.execute("COMMIT")
                return identifiers
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def enqueue_due(
        self,
        *,
        now: datetime,
        misfire_grace_seconds: int = 300,
        limit: int = 100,
        failpoint: Callable[[str], None] | None = None,
    ) -> DueScanResult:
        if misfire_grace_seconds < 0:
            raise ValueError("misfire_grace_seconds 不能为负数")
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        now_utc = _as_utc(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            rows = connection.execute(
                """
                SELECT * FROM scheduled_tasks
                WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ?
                ORDER BY next_run_at, task_id
                LIMIT ?
                """,
                (now_utc.isoformat(), limit),
            ).fetchall()
            queued: list[ScheduledExecution] = []
            missed: list[ScheduledExecution] = []
            for row in rows:
                task = _task_from_row(row)
                assert task.next_run_at is not None
                lateness = (now_utc - task.next_run_at).total_seconds()
                if task.schedule_kind != "every" and lateness > misfire_grace_seconds:
                    missed.append(self._record_missed(connection, task, now_utc))
                    continue
                scheduled_at = (
                    _coalesced_fire_at(task, now_utc)
                    if task.schedule_kind == "every"
                    else task.next_run_at
                )
                queued.append(
                    self._queue_execution(
                        connection,
                        task,
                        now_utc,
                        scheduled_at=scheduled_at,
                        failpoint=failpoint,
                    )
                )
            queued = list(self._pending_direct_executions(connection))
            connection.execute("COMMIT")
            return DueScanResult(tuple(queued), tuple(missed))
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def transition_execution(self, execution_id: str, *, outcome: str, now: datetime) -> str:
        if outcome not in {"running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"不支持的定时执行状态: {outcome}")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM scheduled_executions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if row is None:
                raise KeyError(execution_id)
            current = str(row["state"])
            if current in {"succeeded", "failed", "cancelled"}:
                return current
            connection.execute(
                "UPDATE scheduled_executions SET state = ?, updated_at = ? WHERE execution_id = ?",
                (outcome, _as_utc(now).isoformat(), execution_id),
            )
        return outcome

    def _record_missed(
        self,
        connection: sqlite3.Connection,
        task: ScheduledTask,
        now: datetime,
    ) -> ScheduledExecution:
        assert task.next_run_at is not None
        execution_id = _execution_id(task.task_id, task.next_run_at)
        connection.execute(
            """
            INSERT OR IGNORE INTO scheduled_executions(
                execution_id, task_id, scheduled_at, state, created_at, updated_at
            ) VALUES (?, ?, ?, 'skipped', ?, ?)
            """,
            (
                execution_id,
                task.task_id,
                task.next_run_at.isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        self._finish_or_advance(connection, task, now)
        return ScheduledExecution(execution_id, task.task_id, task.next_run_at, "skipped")

    def _queue_execution(
        self,
        connection: sqlite3.Connection,
        task: ScheduledTask,
        now: datetime,
        *,
        scheduled_at: datetime | None = None,
        failpoint: Callable[[str], None] | None = None,
    ) -> ScheduledExecution:
        assert task.next_run_at is not None
        effective_scheduled_at = scheduled_at or task.next_run_at
        execution_id = _execution_id(task.task_id, effective_scheduled_at)
        session_row = connection.execute(
            "SELECT channel, chat_id FROM sessions WHERE session_key = ?",
            (task.session_key,),
        ).fetchone()
        if session_row is None:
            raise KeyError(task.session_key)
        payload: dict[str, object] = {
            "execution_id": execution_id,
            "execution_mode": task.execution_mode,
            "payload": dict(task.payload),
            "channel": str(session_row["channel"]),
            "chat_id": str(session_row["chat_id"]),
            "scheduled_at": effective_scheduled_at.isoformat(),
            "task_id": task.task_id,
            "task_name": task.name,
            "activity_version": self._activity_version(connection, task.session_key),
        }
        connection.execute(
                """
                INSERT INTO scheduled_executions(
                    execution_id, task_id, scheduled_at, state, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?)
                """,
                (
                    execution_id,
                    task.task_id,
                    effective_scheduled_at.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        self._finish_or_advance(connection, task, now)
        return ScheduledExecution(
                execution_id,
                task.task_id,
                effective_scheduled_at,
                "queued",
                AgentTask(
                    task_id=execution_id,
                    kind="schedule.run",
                    priority=1,
                    session_key=task.session_key,
                    payload=payload,
                    created_at=now,
                ),
        )

    @staticmethod
    def _finish_or_advance(
        connection: sqlite3.Connection,
        task: ScheduledTask,
        now: datetime,
    ) -> None:
        if task.schedule_kind == "every":
            assert task.next_run_at is not None
            next_run_at = advance_every(
                task.schedule_expression,
                timezone_name=task.timezone,
                previous_fire_at=task.next_run_at,
                after=now,
            )
            connection.execute(
                """
                UPDATE scheduled_tasks
                SET next_run_at = ?, version = version + 1, updated_at = ?
                WHERE task_id = ?
                """,
                (next_run_at.isoformat(), now.isoformat(), task.task_id),
            )
        else:
            connection.execute(
                """
                UPDATE scheduled_tasks
                SET enabled = 0, next_run_at = NULL, version = version + 1, updated_at = ?
                WHERE task_id = ?
                """,
                (now.isoformat(), task.task_id),
            )

    def _pending_direct_executions(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[ScheduledExecution, ...]:
        rows = connection.execute(
            """
            SELECT execution.execution_id, execution.task_id,
                   execution.scheduled_at, execution.state, execution.created_at,
                   task.session_key, task.execution_mode, task.payload_json,
                   task.schedule_json, session.channel, session.chat_id,
                   COALESCE(activity.activity_version, 0) AS activity_version
            FROM scheduled_executions AS execution
            JOIN scheduled_tasks AS task ON task.task_id = execution.task_id
            JOIN sessions AS session ON session.session_key = task.session_key
            LEFT JOIN session_activity AS activity
                   ON activity.session_key = task.session_key
            WHERE execution.state = 'queued'
            ORDER BY execution.scheduled_at, execution.execution_id
            """
        ).fetchall()
        pending: list[ScheduledExecution] = []
        for row in rows:
            schedule = json.loads(str(row["schedule_json"]))
            task_payload = json.loads(str(row["payload_json"]))
            scheduled_at = _required_datetime(row["scheduled_at"])
            payload = {
                "execution_id": str(row["execution_id"]),
                "execution_mode": str(row["execution_mode"]),
                "payload": task_payload,
                "channel": str(row["channel"]),
                "chat_id": str(row["chat_id"]),
                "scheduled_at": scheduled_at.isoformat(),
                "task_id": str(row["task_id"]),
                "task_name": schedule.get("name"),
                "activity_version": int(row["activity_version"]),
            }
            pending.append(
                ScheduledExecution(
                    str(row["execution_id"]),
                    str(row["task_id"]),
                    scheduled_at,
                    str(row["state"]),
                    AgentTask(
                        task_id=str(row["execution_id"]),
                        kind="schedule.run",
                        priority=1,
                        session_key=str(row["session_key"]),
                        payload=payload,
                        created_at=_required_datetime(row["created_at"]),
                    ),
                )
            )
        return tuple(pending)

    def _connect(self) -> sqlite3.Connection:
        return connect_database(self.database, busy_timeout_seconds=self.busy_timeout_seconds)

    @staticmethod
    def _activity_version(connection: sqlite3.Connection, session_key: str) -> int:
        row = connection.execute(
            "SELECT activity_version FROM session_activity WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        return 0 if row is None else int(row["activity_version"])


def _task_from_row(row: sqlite3.Row) -> ScheduledTask:
    schedule = json.loads(str(row["schedule_json"]))
    payload = json.loads(str(row["payload_json"]))
    return ScheduledTask(
        task_id=str(row["task_id"]),
        session_key=str(row["session_key"]),
        schedule_kind=cast(ScheduleKind, str(row["schedule_kind"])),
        schedule_expression=str(schedule["expression"]),
        execution_mode=cast(ExecutionMode, str(row["execution_mode"])),
        payload=payload,
        next_run_at=_parse_datetime(row["next_run_at"]),
        timezone=str(schedule.get("timezone") or "Asia/Shanghai"),
        name=cast(str | None, schedule.get("name")),
        enabled=bool(row["enabled"]),
        version=int(row["version"]),
        created_at=_required_datetime(row["created_at"]),
        updated_at=_required_datetime(row["updated_at"]),
    )


def _execution_id(task_id: str, scheduled_at: datetime) -> str:
    return _stable_id("execution", f"{task_id}:{scheduled_at.isoformat()}")


def _coalesced_fire_at(task: ScheduledTask, now: datetime) -> datetime:
    """把积压的 every 任务折叠为最近一个应执行窗口。"""
    assert task.next_run_at is not None
    if is_cron_expr(task.schedule_expression):
        # APScheduler 3 只提供向前搜索；积压 cron 以本次扫描时间代表合并窗口。
        return now
    interval = parse_duration(task.schedule_expression)
    elapsed = now - task.next_run_at
    return (task.next_run_at + interval * (elapsed // interval)).astimezone(UTC)


def _stable_id(kind: str, identity: str) -> str:
    return f"{kind}-{uuid5(NAMESPACE_URL, f'memopilot:{kind}:{identity}').hex}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("时间必须包含时区")
    return value.astimezone(UTC)


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return _required_datetime(value)


def _required_datetime(value: object) -> datetime:
    return _as_utc(datetime.fromisoformat(str(value)))


__all__ = ["ScheduleRepository"]
