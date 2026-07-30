"""会话、消息与后台提交权的轻量 SQLite 仓储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.persistence.migrations import connect_database
from memopilot.tasks.background import BackgroundTask


class LostLeaseError(RuntimeError):
    """当前执行者已经失去会话提交权。"""


class StaleActivityError(LostLeaseError):
    """用户活动已变化，当前主动任务应结束并等待下一轮。"""


class MultiplePrivateSessionsError(RuntimeError):
    """单用户版本检测到多个飞书私聊目标。"""


class FenceToken(Protocol):
    @property
    def session_key(self) -> str: ...

    @property
    def owner_id(self) -> str: ...

    @property
    def epoch(self) -> int: ...


class TurnMessage(Protocol):
    channel: str
    chat_id: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any]

    @property
    def session_key(self) -> str: ...


@dataclass(frozen=True, slots=True)
class SessionIdentityRecord:
    identity_kind: str
    identity_value: str
    session_key: str
    chat_id: str


@dataclass(frozen=True, slots=True)
class PrivateSessionTarget:
    session_key: str
    channel: str
    chat_id: str
    activity_version: int


@dataclass(frozen=True, slots=True)
class MessageRecord:
    message_id: str
    session_key: str
    role: str
    content: str
    turn_id: str
    session_position: int
    created_at: str
    tool_chain: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class TurnCommitResult:
    assistant_content: str
    media: tuple[str, ...]
    tool_chain: tuple[dict[str, object], ...]
    inserted: bool
    background_tasks: tuple[BackgroundTask, ...]


class OperationalRepository:
    """只保存跨重启仍有业务意义的数据。"""

    _COUNTABLE_TABLES = frozenset(
        {
            "sessions",
            "session_activity",
            "inbound_events",
            "session_identities",
            "messages",
            "scheduled_tasks",
            "scheduled_executions",
            "consolidation_manifests",
        }
    )

    def __init__(self, database: Path, *, busy_timeout_seconds: float = 5) -> None:
        self.database = Path(database)
        self.busy_timeout_seconds = busy_timeout_seconds

    def record_inbound_activity(self, message: TurnMessage) -> int:
        timestamp = message.timestamp
        metadata = message.metadata
        now_text = _utc_iso(timestamp)
        message_id = str(metadata.get("message_id") or "").strip()
        if not message_id:
            message_id = _stable_id(
                "inbound",
                f"{message.session_key}:{now_text}:{message.content}",
            )
        event_id = str(metadata.get("event_id") or message_id)
        session_key = message.session_key
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._upsert_session(
                connection,
                session_key=session_key,
                channel=message.channel,
                chat_id=message.chat_id,
                now_text=now_text,
            )
            existing = connection.execute(
                "SELECT 1 FROM inbound_events WHERE event_id = ? OR message_id = ?",
                (event_id, message_id),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO inbound_events(
                        event_id, message_id, session_key, payload_json, received_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        message_id,
                        session_key,
                        json.dumps(
                            {"content": message.content},
                            ensure_ascii=False,
                        ),
                        now_text,
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
                    (session_key, now_text, now_text),
                )
            row = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO session_activity(session_key, activity_version, updated_at) "
                    "VALUES (?, 0, ?)",
                    (session_key, now_text),
                )
                activity_version = 0
            else:
                activity_version = int(row["activity_version"])
            connection.execute("COMMIT")
            return activity_version
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def commit_turn(
        self,
        message: TurnMessage,
        *,
        assistant_content: str | None = None,
        assistant_media: tuple[str, ...] = (),
        assistant_tool_chain: tuple[dict[str, object], ...] = (),
        cited_memory_ids: tuple[str, ...] = (),
        explicitly_memorized_ids: tuple[str, ...] = (),
        lease: FenceToken | None = None,
    ) -> TurnCommitResult | None:
        content = message.content
        if not content.strip() or (
            assistant_content is not None and not assistant_content.strip()
        ):
            raise ValueError("Turn 的用户消息和助手回复不能为空")
        timestamp = message.timestamp
        metadata = message.metadata
        now_text = _utc_iso(timestamp)
        session_key = message.session_key
        turn_id = _stable_id(
            "turn",
            str(metadata.get("message_id") or f"{session_key}:{now_text}"),
        )
        user_id = _stable_id("message", f"{turn_id}:user")
        assistant_id = _stable_id("message", f"{turn_id}:assistant")
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            if lease is not None:
                self.require_current_fence(connection, lease)
            self._upsert_session(
                connection,
                session_key=session_key,
                channel=message.channel,
                chat_id=message.chat_id,
                now_text=now_text,
            )
            existing = connection.execute(
                "SELECT role, content, media_json, tool_chain_json FROM messages "
                "WHERE turn_id = ? ORDER BY turn_position",
                (turn_id,),
            ).fetchall()
            inserted = not existing
            if existing:
                if (
                    len(existing) != 2
                    or (str(existing[0]["role"]), str(existing[0]["content"]))
                    != ("user", content)
                ):
                    raise ValueError("同一 Turn 不能以不同内容重复提交")
                persisted_assistant = str(existing[1]["content"])
                persisted_media = _parse_media_json(existing[1]["media_json"])
                persisted_tool_chain = _parse_tool_chain_json(existing[1]["tool_chain_json"])
                if assistant_content is not None and (
                    str(existing[1]["role"]),
                    persisted_assistant,
                ) != (
                    "assistant",
                    assistant_content,
                ):
                    raise ValueError("同一 Turn 不能以不同内容重复提交")
                if (
                    assistant_content is not None
                    and assistant_media
                    and persisted_media != assistant_media
                ):
                    raise ValueError("同一 Turn 不能以不同媒体重复提交")
                if assistant_tool_chain and persisted_tool_chain != assistant_tool_chain:
                    raise ValueError("同一 Turn 不能以不同工具链重复提交")
            elif assistant_content is None:
                connection.execute("COMMIT")
                return None
            else:
                persisted_assistant = assistant_content
                persisted_media = assistant_media
                persisted_tool_chain = assistant_tool_chain
                position = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(session_position), 0) + 1 "
                        "FROM messages WHERE session_key = ?",
                        (session_key,),
                    ).fetchone()[0]
                )
                connection.executemany(
                    """
                    INSERT INTO messages(
                        message_id, session_key, role, content, turn_id,
                        turn_position, created_at, session_position, media_json, tool_chain_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            user_id,
                            session_key,
                            "user",
                            content,
                            turn_id,
                            0,
                            now_text,
                            position,
                            "[]",
                            "[]",
                        ),
                        (
                            assistant_id,
                            session_key,
                            "assistant",
                            assistant_content,
                            turn_id,
                            1,
                            now_text,
                            position + 1,
                            json.dumps(assistant_media, ensure_ascii=False),
                            json.dumps(assistant_tool_chain, ensure_ascii=False),
                        ),
                    ),
                )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

        tasks = [
            BackgroundTask(
                _stable_id("task", f"consolidate:{turn_id}"),
                "memory.consolidate",
                3,
                session_key,
                {"trigger_turn_id": turn_id, "last_message_id": assistant_id},
                timestamp,
            ),
            BackgroundTask(
                _stable_id("task", f"post-response:{turn_id}"),
                "memory.post_response",
                3,
                session_key,
                {
                    "turn_id": turn_id,
                    "protected_ids": list(
                        dict.fromkeys(
                            item.strip()
                            for item in explicitly_memorized_ids
                            if item.strip()
                        )
                    ),
                },
                timestamp,
            ),
        ]
        cited = tuple(dict.fromkeys(item.strip() for item in cited_memory_ids if item.strip()))
        if cited:
            tasks.append(
                BackgroundTask(
                    _stable_id("task", f"memory-reinforce:{turn_id}"),
                    "memory.reinforce",
                    3,
                    session_key,
                    {"usage_ref": f"turn:{turn_id}", "item_ids": list(cited)},
                    timestamp,
                )
            )
        return TurnCommitResult(
            persisted_assistant,
            persisted_media,
            persisted_tool_chain,
            inserted,
            tuple(tasks),
        )

    def list_recent_messages(
        self,
        session_key: str,
        *,
        limit: int,
        before: datetime | None = None,
    ) -> tuple[MessageRecord, ...]:
        if limit < 1:
            return ()
        before_text = None if before is None else _utc_iso(before)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_id, session_key, role, content, turn_id,
                       session_position, created_at, tool_chain_json
                FROM (
                    SELECT message_id, session_key, role, content, turn_id,
                           session_position, created_at, tool_chain_json
                    FROM messages
                    WHERE session_key = ? AND session_position IS NOT NULL
                      AND (? IS NULL OR julianday(created_at) <= julianday(?))
                    ORDER BY session_position DESC
                    LIMIT ?
                )
                ORDER BY session_position
                """,
                (session_key, before_text, before_text, limit),
            ).fetchall()
        return tuple(
            MessageRecord(
                str(row["message_id"]),
                str(row["session_key"]),
                str(row["role"]),
                str(row["content"]),
                str(row["turn_id"]),
                int(row["session_position"]),
                str(row["created_at"]),
                _parse_tool_chain_json(row["tool_chain_json"]),
            )
            for row in rows
        )

    def memory_status(
        self,
        session_key: str,
    ) -> tuple[int, int, int, str]:
        with self._connect() as connection:
            session = connection.execute(
                "SELECT last_consolidated_position FROM sessions "
                "WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            last_position = (
                int(session["last_consolidated_position"])
                if session is not None
                else 0
            )
            totals = connection.execute(
                """
                SELECT COUNT(*) AS total_messages,
                       SUM(CASE WHEN role = 'user' AND session_position > ?
                           THEN 1 ELSE 0 END) AS pending_user
                FROM messages
                WHERE session_key = ?
                """,
                (last_position, session_key),
            ).fetchone()
            last_user = connection.execute(
                """
                SELECT content FROM messages
                WHERE session_key = ? AND role = 'user'
                  AND session_position <= ?
                ORDER BY session_position DESC
                LIMIT 1
                """,
                (session_key, last_position),
            ).fetchone()
        return (
            last_position,
            int(totals["total_messages"] or 0),
            int(totals["pending_user"] or 0),
            str(last_user["content"]) if last_user is not None else "",
        )

    def undo_last_turn(
        self,
        session_key: str,
    ) -> tuple[int, int, int] | None:
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            target = connection.execute(
                """
                SELECT user.message_id AS user_id,
                       assistant.message_id AS assistant_id
                FROM messages AS user
                JOIN messages AS assistant
                  ON assistant.turn_id = user.turn_id
                 AND assistant.turn_position = 1
                 AND assistant.role = 'assistant'
                WHERE user.session_key = ?
                  AND user.turn_position = 0
                  AND user.role = 'user'
                ORDER BY assistant.session_position DESC
                LIMIT 1
                """,
                (session_key,),
            ).fetchone()
            session = connection.execute(
                "SELECT last_consolidated_position FROM sessions "
                "WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if target is None or session is None:
                connection.execute("COMMIT")
                return None
            old_cursor = int(session["last_consolidated_position"])
            deleted = connection.execute(
                "DELETE FROM messages WHERE message_id IN (?, ?)",
                (str(target["user_id"]), str(target["assistant_id"])),
            ).rowcount
            if deleted != 2:
                raise RuntimeError("撤销目标不再完整")
            remaining = int(
                connection.execute(
                    "SELECT COALESCE(MAX(session_position), 0) FROM messages "
                    "WHERE session_key = ?",
                    (session_key,),
                ).fetchone()[0]
            )
            new_cursor = min(old_cursor, remaining)
            connection.execute(
                "UPDATE sessions SET last_consolidated_position = ?, "
                "updated_at = ? WHERE session_key = ?",
                (new_cursor, datetime.now(UTC).isoformat(), session_key),
            )
            connection.execute("COMMIT")
            return deleted, old_cursor, new_cursor
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def remember_session_identities(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        identities: Mapping[str, str],
        now: datetime,
    ) -> None:
        allowed = {"open_id", "user_id", "union_id"}
        normalized = {
            kind: str(value).strip()
            for kind, value in identities.items()
            if kind in allowed and str(value).strip()
        }
        if not normalized:
            return
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self._upsert_session(
                connection,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                now_text=now_text,
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
                FROM session_identities WHERE channel = ?
                ORDER BY identity_kind, identity_value
                """,
                (channel,),
            ).fetchall()
        return tuple(
            SessionIdentityRecord(
                str(row["identity_kind"]),
                str(row["identity_value"]),
                str(row["session_key"]),
                str(row["chat_id"]),
            )
            for row in rows
        )

    def get_single_private_session(self) -> PrivateSessionTarget | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session.session_key, session.channel, session.chat_id,
                       COALESCE(activity.activity_version, 0) AS activity_version
                FROM sessions AS session
                LEFT JOIN session_activity AS activity USING(session_key)
                WHERE session.channel = 'feishu'
                ORDER BY session.created_at, session.session_key
                """
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise MultiplePrivateSessionsError(
                "MemoPilot 当前只支持一个飞书私聊，检测到多个会话目标"
            )
        row = rows[0]
        return PrivateSessionTarget(
            str(row["session_key"]),
            str(row["channel"]),
            str(row["chat_id"]),
            int(row["activity_version"]),
        )

    def ensure_system_session(self, session_key: str, *, chat_id: str, now: datetime) -> None:
        now_text = _utc_iso(now)
        with self._connect() as connection:
            self._upsert_session(
                connection,
                session_key=session_key,
                channel="system",
                chat_id=chat_id,
                now_text=now_text,
            )
            connection.execute(
                "INSERT OR IGNORE INTO session_activity("
                "session_key, activity_version, updated_at) VALUES (?, 0, ?)",
                (session_key, now_text),
            )

    def get_activity_version(self, session_key: str) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return None if row is None else int(row["activity_version"])

    def allocate_fence(self, session_key: str, *, owner_id: str, now: datetime) -> int:
        now_text = _utc_iso(now)
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            if connection.execute(
                "SELECT 1 FROM sessions WHERE session_key = ?",
                (session_key,),
            ).fetchone() is None:
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
                UPDATE session_fences SET heartbeat_at = ?, updated_at = ?
                WHERE session_key = ? AND owner_id = ? AND current_epoch = ?
                """,
                (now_text, now_text, lease.session_key, lease.owner_id, lease.epoch),
            ).rowcount
        return changed == 1

    @staticmethod
    def require_current_fence(connection: sqlite3.Connection, lease: FenceToken) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM session_fences
            WHERE session_key = ? AND owner_id = ? AND current_epoch = ?
            """,
            (lease.session_key, lease.owner_id, lease.epoch),
        ).fetchone()
        if row is None:
            raise LostLeaseError(
                f"会话 {lease.session_key} 的 owner/epoch 已失效: "
                f"{lease.owner_id}/{lease.epoch}"
            )

    def assert_current_fence(self, lease: FenceToken) -> None:
        with self._connect() as connection:
            self.require_current_fence(connection, lease)

    def assert_current_fence_and_activity(
        self,
        lease: FenceToken,
        *,
        expected_activity_version: int,
    ) -> None:
        with self._connect() as connection:
            self.require_current_fence(connection, lease)
            row = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (lease.session_key,),
            ).fetchone()
            current = None if row is None else int(row["activity_version"])
            if current != expected_activity_version:
                raise StaleActivityError(
                    f"会话 {lease.session_key} 的 activity_version 已变化: "
                    f"expected={expected_activity_version}, current={current}"
                )

    @contextmanager
    def fenced_write(self, lease: FenceToken) -> Iterator[None]:
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self.require_current_fence(connection, lease)
            yield
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def transition_background_schedule(
        self,
        execution_id: str,
        *,
        lease: FenceToken,
        outcome: str,
        now: datetime,
    ) -> str:
        allowed = {"running", "succeeded", "failed", "cancelled"}
        if outcome not in allowed:
            raise ValueError(f"不支持的定时执行状态: {outcome}")
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            self.require_current_fence(connection, lease)
            row = connection.execute(
                """
                SELECT execution.state
                FROM scheduled_executions AS execution
                JOIN scheduled_tasks AS task ON task.task_id = execution.task_id
                WHERE execution.execution_id = ? AND task.session_key = ?
                """,
                (execution_id, lease.session_key),
            ).fetchone()
            if row is None:
                raise KeyError(execution_id)
            current = str(row["state"])
            terminal = {"succeeded", "failed", "cancelled"}
            if current in terminal:
                if outcome == "running" or outcome == current:
                    connection.execute("COMMIT")
                    return current
                raise RuntimeError(
                    f"定时执行 {execution_id} 已终结为 {current}，不能改为 {outcome}"
                )
            connection.execute(
                "UPDATE scheduled_executions SET state = ?, updated_at = ? "
                "WHERE execution_id = ?",
                (outcome, _utc_iso(now), execution_id),
            )
            connection.execute("COMMIT")
            return outcome
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

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
    def _upsert_session(
        connection: sqlite3.Connection,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        now_text: str,
    ) -> None:
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


def _utc_iso(value: datetime) -> str:
    current = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return current.astimezone(UTC).isoformat()


def _stable_id(namespace: str, value: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"memopilot:{namespace}:{value}"))


def _parse_media_json(value: object) -> tuple[str, ...]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("持久化消息的 media_json 不是合法 JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError("持久化消息的 media_json 必须是列表")
    if any(not isinstance(item, str) or not item.strip() for item in decoded):
        raise ValueError("持久化消息的 media_json 只能包含非空字符串")
    return tuple(decoded)


def _parse_tool_chain_json(value: object) -> tuple[dict[str, object], ...]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("持久化消息的 tool_chain_json 不是合法 JSON") from exc
    if not isinstance(decoded, list) or any(not isinstance(item, dict) for item in decoded):
        raise ValueError("持久化消息的 tool_chain_json 必须是对象列表")
    return tuple(decoded)


__all__ = [
    "FenceToken",
    "LostLeaseError",
    "MessageRecord",
    "MultiplePrivateSessionsError",
    "OperationalRepository",
    "PrivateSessionTarget",
    "SessionIdentityRecord",
    "StaleActivityError",
]
