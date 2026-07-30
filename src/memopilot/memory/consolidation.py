"""消息窗口归档、Manifest 驱动的多文件提交与恢复。"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol, cast
from uuid import NAMESPACE_URL, uuid5

from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.persistence.migrations import connect_database
from memopilot.tasks.operational import FenceToken, OperationalRepository


class ConsolidationExtractor(Protocol):
    async def extract(self, conversation: str) -> dict[str, object]: ...


class RecentContextCompressor(Protocol):
    async def compress(
        self,
        *,
        old_context: str,
        conversation: str,
        recent_turns: str,
        compression_until: str,
    ) -> str: ...


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
        recent_context: RecentContextCompressor | None = None,
        keep_count: int = 12,
        min_new_messages: int = 5,
        recent_turn_count: int,
        failpoint: Callable[[str], None] | None = None,
    ) -> None:
        self.database = database
        self.markdown = markdown
        self.extractor = extractor
        self.recent_context = recent_context
        self.keep_count = max(0, keep_count)
        self.min_new_messages = max(1, min_new_messages)
        if recent_turn_count <= 0:
            raise ValueError("recent_turn_count 必须大于 0")
        self.recent_turn_count = recent_turn_count
        self.failpoint = failpoint

    async def run(
        self,
        session_key: str,
        *,
        assert_current: Callable[[], None] | None = None,
        lease: FenceToken | None = None,
        fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> ConsolidationResult | None:
        guard = assert_current or (lambda: None)
        guard()
        manifest = self._pending_manifest(session_key)
        if manifest is None:
            window = self._select_window(session_key)
            if window is None:
                self._refresh_recent_turns(session_key)
                return None
            output = await self.extractor.extract(_format_conversation(window))
            if self.recent_context is not None:
                output = dict(output)
                artifacts = _validated_artifacts(output.get("artifacts"))
                artifacts["RECENT_CONTEXT.md"] = await self.recent_context.compress(
                    old_context=self.markdown.read("RECENT_CONTEXT.md"),
                    conversation=_format_conversation(window),
                    recent_turns=self._recent_turns(session_key),
                    compression_until=str(window[-1]["created_at"]),
                )
                output["artifacts"] = artifacts
            guard()
            manifest = self._create_manifest(session_key, window, output)
        result = self._write_and_commit(
            manifest,
            assert_current=guard,
            lease=lease,
            fenced_write=fenced_write or nullcontext,
        )
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

    def _recent_turns(self, session_key: str) -> str:
        recent_count = min(self.recent_turn_count, self.keep_count)
        with connect_database(self.database) as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE session_key = ? "
                "ORDER BY session_position DESC LIMIT ?",
                (session_key, recent_count),
            ).fetchall()
        lines: list[str] = []
        for row in reversed(rows):
            role = str(row["role"])
            content = str(row["content"]).strip()
            if role == "user" and content:
                lines.append(f"[user] {content}")
            elif role == "assistant" and content:
                lines.append(f"[a-preview] {content[:60]}")
        return "\n".join(lines)

    def _refresh_recent_turns(self, session_key: str) -> None:
        current = self.markdown.read("RECENT_CONTEXT.md")
        self.markdown.replace(
            "RECENT_CONTEXT.md",
            _replace_recent_turns_block(current, self._recent_turns(session_key)),
        )

    def _create_manifest(
        self,
        session_key: str,
        window: list[sqlite3.Row],
        output: dict[str, object],
    ) -> sqlite3.Row:
        history_entries = _normalize_history_entries(output.get("history_entries"))
        pending_items = _format_pending_items(output.get("pending_items"))
        artifacts = _validated_artifacts(output.get("artifacts"))
        if history_entries:
            artifacts["HISTORY.md"] = "\n".join(str(entry["summary"]) for entry in history_entries)
        if pending_items:
            artifacts["PENDING.md"] = pending_items
        artifacts.update(
            _journal_artifacts_from_history_entries(
                [str(entry["summary"]) for entry in history_entries]
            )
        )
        first = window[0]
        last = window[-1]
        consolidation_id = str(
            uuid5(
                NAMESPACE_URL,
                f"memopilot:consolidation:{session_key}:{first['message_id']}:{last['message_id']}",
            )
        )
        hashes = {name: self.markdown.content_hash(content) for name, content in artifacts.items()}
        states = {name: "pending" for name in artifacts}
        persisted_output = dict(output)
        persisted_output["_conversation"] = _format_conversation(window)
        if history_entries:
            persisted_output["history_entries"] = history_entries
        if pending_items:
            persisted_output["pending_items"] = pending_items.splitlines()
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

    def _write_and_commit(
        self,
        manifest: sqlite3.Row,
        *,
        assert_current: Callable[[], None],
        lease: FenceToken | None,
        fenced_write: Callable[[], AbstractContextManager[None]],
    ) -> ConsolidationResult:
        consolidation_id = str(manifest["consolidation_id"])
        output = _json_object(manifest["model_output_json"])
        artifacts = _validated_artifacts(output.get("artifacts"))
        hashes = {str(k): str(v) for k, v in _json_object(manifest["artifact_hashes_json"]).items()}
        states = {str(k): str(v) for k, v in _json_object(manifest["artifact_states_json"]).items()}
        assert_current()
        self._set_manifest_state(consolidation_id, "writing", attempts_delta=1)
        for name, content in artifacts.items():
            content_hash = hashes[name]
            if states.get(name) != "written" or not self.markdown.contains_artifact(
                name, consolidation_id, content_hash
            ):
                assert_current()
                with fenced_write():
                    self.markdown.append_artifact(
                        name,
                        consolidation_id=consolidation_id,
                        content=content,
                        content_hash=content_hash,
                    )
                states[name] = "written"
                assert_current()
                self._save_artifact_states(consolidation_id, states)
            if self.failpoint is not None:
                self.failpoint(f"after_artifact:{name}")
        if not all(
            self.markdown.contains_artifact(name, consolidation_id, hashes[name])
            for name in artifacts
        ):
            raise RuntimeError("归档文件校验失败，不允许发布向量任务")
        window = _json_object(output.get("_window"))
        assert_current()
        self._commit_manifest(
            consolidation_id,
            session_key=str(manifest["session_key"]),
            last_position=int(str(window["last_position"])),
            lease=lease,
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
        self,
        consolidation_id: str,
        *,
        session_key: str,
        last_position: int,
        lease: FenceToken | None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        connection = connect_database(self.database)
        connection.execute("BEGIN IMMEDIATE")
        try:
            if lease is not None:
                OperationalRepository.require_current_fence(connection, lease)
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
    allowed = {"HISTORY.md", "PENDING.md", "RECENT_CONTEXT.md"}
    result = {str(name): str(content).strip() for name, content in value.items() if content}
    unsupported = {
        name
        for name in result
        if name not in allowed and not re.fullmatch(r"journal/\d{4}-\d{2}-\d{2}\.md", name)
    }
    if unsupported:
        raise ValueError(f"Consolidation 返回了不支持的文件: {sorted(unsupported)}")
    for name in result:
        if name.startswith("journal/"):
            try:
                date.fromisoformat(Path(name).stem)
            except ValueError:
                raise ValueError(f"Consolidation journal 日期无效: {name}") from None
    return result


def _journal_artifacts_from_history_entries(entries: list[str]) -> dict[str, str]:
    by_date: dict[str, list[str]] = {}
    for entry in entries:
        match = re.match(r"^\[(\d{4}-\d{2}-\d{2})(?:\s+\d{2}:\d{2})?]", entry)
        if match is None:
            continue
        try:
            day = date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            raise ValueError(f"history_entry 日期无效: {match.group(1)}") from None
        by_date.setdefault(day, []).append(entry)
    return {f"journal/{day}.md": "\n".join(summaries) for day, summaries in by_date.items()}


def _normalize_history_entries(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in value:
        if isinstance(raw, str):
            summary = raw.strip()
            weight = 0
        elif isinstance(raw, dict):
            summary = str(raw.get("summary") or "").strip()
            raw_weight = raw.get("emotional_weight", 0)
            try:
                weight = max(0, min(10, int(raw_weight)))
            except (TypeError, ValueError):
                weight = 0
        else:
            continue
        if not summary or summary in seen:
            continue
        seen.add(summary)
        result.append({"summary": summary, "emotional_weight": weight})
    return result


def _format_pending_items(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    lines: list[str] = []
    for item in value:
        if isinstance(item, dict):
            tag = str(item.get("tag") or "").strip()
            content = str(item.get("content") or "").strip()
            if tag and content:
                lines.append(f"- [{tag}] {content}")
        else:
            text = str(item).strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def _replace_recent_turns_block(old_context: str, recent_turns: str) -> str:
    marker = "\n## Recent Turns\n"
    block = "## Recent Turns\n<!-- a-preview = assistant reply preview only -->\n" + (
        recent_turns.strip() or "- none"
    )
    current = old_context.strip()
    if marker in current:
        return current.split(marker, 1)[0].rstrip() + "\n\n" + block
    if current:
        return current + "\n\n" + block
    return "# Recent Context\n\n## Compression\n- none\n\n" + block


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    loaded = json.loads(str(value))
    if not isinstance(loaded, dict):
        raise ValueError("期望 JSON 对象")
    return {str(key): item for key, item in loaded.items()}


__all__ = ["ConsolidationResult", "ConsolidationService"]
