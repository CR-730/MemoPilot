"""消息窗口归档、Manifest 驱动的多文件提交与恢复。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast
from uuid import NAMESPACE_URL, uuid5

from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.persistence.migrations import connect_database


class ConsolidationExtractor(Protocol):
    async def extract(self, conversation: str) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    consolidation_id: str
    first_position: int
    last_position: int
    message_count: int


class ConsolidationService:
    def __init__(
        self,
        database: Path,
        markdown: MarkdownMemoryStore,
        extractor: ConsolidationExtractor,
        *,
        keep_count: int = 12,
        min_new_messages: int = 5,
        failpoint: Callable[[str], None] | None = None,
    ) -> None:
        self.database = database
        self.markdown = markdown
        self.extractor = extractor
        self.keep_count = max(0, keep_count)
        self.min_new_messages = max(1, min_new_messages)
        self.failpoint = failpoint

    async def run(self, session_key: str) -> ConsolidationResult | None:
        manifest = self._pending_manifest(session_key)
        if manifest is None:
            window = self._select_window(session_key)
            if window is None:
                return None
            output = await self.extractor.extract(_format_conversation(window))
            manifest = self._create_manifest(session_key, window, output)
        result = self._write_and_commit(manifest)
        return result

    def _select_window(self, session_key: str) -> list[sqlite3.Row] | None:
        with connect_database(self.database) as connection:
            session = connection.execute(
                "SELECT last_consolidated_position FROM sessions WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if session is None:
                raise KeyError(session_key)
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_key = ? AND session_position > ? "
                "ORDER BY session_position",
                (session_key, int(session["last_consolidated_position"])),
            ).fetchall()
        consolidate_count = len(rows) - self.keep_count
        if consolidate_count < self.min_new_messages:
            return None
        return list(rows[:consolidate_count])

    def _pending_manifest(self, session_key: str) -> sqlite3.Row | None:
        with connect_database(self.database) as connection:
            row = connection.execute(
                "SELECT * FROM consolidation_manifests "
                "WHERE session_key = ? AND state IN ('pending', 'writing', 'failed') "
                "ORDER BY created_at LIMIT 1",
                (session_key,),
            ).fetchone()
        return cast(sqlite3.Row | None, row)

    def _create_manifest(
        self,
        session_key: str,
        window: list[sqlite3.Row],
        output: dict[str, object],
    ) -> sqlite3.Row:
        artifacts = _validated_artifacts(output.get("artifacts"))
        first = window[0]
        last = window[-1]
        consolidation_id = str(
            uuid5(
                NAMESPACE_URL,
                f"memopilot:consolidation:{session_key}:{first['message_id']}:{last['message_id']}",
            )
        )
        hashes = {
            name: self.markdown.content_hash(content) for name, content in artifacts.items()
        }
        states = {name: "pending" for name in artifacts}
        persisted_output = dict(output)
        persisted_output["artifacts"] = artifacts
        persisted_output["_window"] = {
            "first_position": int(first["session_position"]),
            "last_position": int(last["session_position"]),
            "message_count": len(window),
        }
        now = datetime.now(UTC).isoformat()
        with connect_database(self.database) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO consolidation_manifests("
                "consolidation_id, session_key, first_message_id, last_message_id, "
                "artifact_hashes_json, model_output_json, state, attempts, created_at, updated_at, "
                "artifact_states_json) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)",
                (
                    consolidation_id,
                    session_key,
                    str(first["message_id"]),
                    str(last["message_id"]),
                    json.dumps(hashes, ensure_ascii=False, sort_keys=True),
                    json.dumps(persisted_output, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                    json.dumps(states, ensure_ascii=False, sort_keys=True),
                ),
            )
            row = connection.execute(
                "SELECT * FROM consolidation_manifests WHERE consolidation_id = ?",
                (consolidation_id,),
            ).fetchone()
        assert row is not None
        return cast(sqlite3.Row, row)

    def _write_and_commit(self, manifest: sqlite3.Row) -> ConsolidationResult:
        consolidation_id = str(manifest["consolidation_id"])
        output = _json_object(manifest["model_output_json"])
        artifacts = _validated_artifacts(output.get("artifacts"))
        hashes = {str(k): str(v) for k, v in _json_object(manifest["artifact_hashes_json"]).items()}
        states = {
            str(k): str(v) for k, v in _json_object(manifest["artifact_states_json"]).items()
        }
        self._set_manifest_state(consolidation_id, "writing", attempts_delta=1)
        for name, content in artifacts.items():
            content_hash = hashes[name]
            if states.get(name) != "written" or not self.markdown.contains_artifact(
                name, consolidation_id, content_hash
            ):
                self.markdown.append_artifact(
                    name,
                    consolidation_id=consolidation_id,
                    content=content,
                    content_hash=content_hash,
                )
                states[name] = "written"
                self._save_artifact_states(consolidation_id, states)
            if self.failpoint is not None:
                self.failpoint(f"after_artifact:{name}")
        if not all(
            self.markdown.contains_artifact(name, consolidation_id, hashes[name])
            for name in artifacts
        ):
            raise RuntimeError("归档文件校验失败，不允许发布向量任务")
        window = _json_object(output.get("_window"))
        self._commit_manifest(
            consolidation_id,
            session_key=str(manifest["session_key"]),
            last_position=int(str(window["last_position"])),
        )
        return ConsolidationResult(
            consolidation_id=consolidation_id,
            first_position=int(str(window["first_position"])),
            last_position=int(str(window["last_position"])),
            message_count=int(str(window["message_count"])),
        )

    def _set_manifest_state(
        self, consolidation_id: str, state: str, *, attempts_delta: int = 0
    ) -> None:
        with connect_database(self.database) as connection:
            connection.execute(
                "UPDATE consolidation_manifests SET state = ?, attempts = attempts + ?, "
                "updated_at = ?, last_error = NULL WHERE consolidation_id = ?",
                (state, attempts_delta, datetime.now(UTC).isoformat(), consolidation_id),
            )

    def _save_artifact_states(self, consolidation_id: str, states: dict[str, str]) -> None:
        with connect_database(self.database) as connection:
            connection.execute(
                "UPDATE consolidation_manifests SET artifact_states_json = ?, updated_at = ? "
                "WHERE consolidation_id = ?",
                (
                    json.dumps(states, ensure_ascii=False, sort_keys=True),
                    datetime.now(UTC).isoformat(),
                    consolidation_id,
                ),
            )

    def _commit_manifest(
        self, consolidation_id: str, *, session_key: str, last_position: int
    ) -> None:
        now = datetime.now(UTC).isoformat()
        job_id = str(uuid5(NAMESPACE_URL, f"memopilot:vectorize:{consolidation_id}"))
        outbox_id = str(uuid5(NAMESPACE_URL, f"memopilot:outbox:{job_id}"))
        connection = connect_database(self.database)
        connection.execute("BEGIN IMMEDIATE")
        try:
            activity = connection.execute(
                "SELECT activity_version FROM session_activity WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            activity_version = int(activity[0]) if activity is not None else 0
            payload = {
                "job_id": job_id,
                "kind": "memory.vectorize",
                "session_key": session_key,
                "priority": 3,
                "consolidation_id": consolidation_id,
            }
            connection.execute(
                "INSERT OR IGNORE INTO agent_jobs("
                "job_id, kind, priority, session_key, idempotency_key, state, activity_version, "
                "payload_json, created_at, updated_at) VALUES (?, 'memory.vectorize', 3, ?, ?, "
                "'queued', ?, ?, ?, ?)",
                (
                    job_id,
                    session_key,
                    f"vectorize:{consolidation_id}",
                    activity_version,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO outbox_events("
                "outbox_id, event_type, aggregate_id, payload_json, idempotency_key, state, "
                "next_attempt_at, created_at, updated_at) VALUES (?, 'job.ready', ?, ?, ?, "
                "'pending', ?, ?, ?)",
                (
                    outbox_id,
                    job_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    f"publish-job:{job_id}",
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE sessions SET last_consolidated_position = MAX("
                "last_consolidated_position, ?), updated_at = ? WHERE session_key = ?",
                (last_position, now, session_key),
            )
            connection.execute(
                "UPDATE consolidation_manifests SET state = 'committed', committed_at = ?, "
                "updated_at = ?, last_error = NULL WHERE consolidation_id = ?",
                (now, now, consolidation_id),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def _format_conversation(rows: list[sqlite3.Row]) -> str:
    return "\n".join(
        f"[{row['created_at']}] {str(row['role']).upper()}: {row['content']}" for row in rows
    )


def _validated_artifacts(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    allowed = {"HISTORY.md", "PENDING.md", "CONTEXT.md"}
    result = {str(name): str(content).strip() for name, content in value.items() if content}
    unsupported = set(result) - allowed
    if unsupported:
        raise ValueError(f"Consolidation 返回了不支持的文件: {sorted(unsupported)}")
    return result


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    loaded = json.loads(str(value))
    if not isinstance(loaded, dict):
        raise ValueError("期望 JSON 对象")
    return {str(key): item for key, item in loaded.items()}


__all__ = ["ConsolidationResult", "ConsolidationService"]
