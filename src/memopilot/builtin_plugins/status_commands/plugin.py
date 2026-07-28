from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import BeforeTurnCtx, BeforeTurnInput

_MEMORY_COMMANDS = {"/memorystatus", "/memory_status", "/compact_status"}
_CACHE_COMMANDS = {"/kvcache", "/cache_status"}
_BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class StatusCommandModule:
    slot = "status_commands.handle"
    requires = ("before_turn.build_ctx", "session:ctx")
    produces = ("session:ctx",)

    def __init__(self, repository: object, db_path: Path | None) -> None:
        self.repository = repository
        self.db_path = db_path

    async def run(self, frame: Any) -> Any:
        state = frame.input
        ctx = frame.slots.get("session:ctx")
        if not isinstance(state, BeforeTurnInput) or not isinstance(ctx, BeforeTurnCtx):
            return frame
        command = _normalize_command(state.content)
        if command in _MEMORY_COMMANDS:
            reply = _memory_status(self.repository, state.session_key)
        elif command in _CACHE_COMMANDS:
            reply = _cache_status(self.db_path, state.session_key, state.content)
        else:
            return frame
        ctx.abort = True
        ctx.abort_reply = reply
        return frame


class StatusCommandsPlugin(Plugin):
    name = "status_commands"

    def before_turn_modules(self) -> list[object]:
        db_path = (
            self.context.workspace / "observe" / "observe.db"
            if self.context.workspace is not None
            else None
        )
        return [
            StatusCommandModule(
                self.context.session_manager,
                db_path,
            )
        ]


def _normalize_command(content: str) -> str:
    parts = content.strip().split(maxsplit=1)
    if not parts:
        return ""
    return parts[0].lower().split("@", 1)[0]


def _memory_status(repository: object, session_key: str) -> str:
    read = getattr(repository, "memory_status", None)
    if not callable(read):
        return "记忆整理状态查询不可用。"
    try:
        last_position, total_messages, pending, last_content = read(session_key)
    except Exception:
        return "记忆整理状态查询失败。"
    last_content = str(last_content).strip()

    lines = ["🧠 记忆整理状态："]
    if int(last_position) <= 0 or not last_content:
        lines.append("当前会话还没有完成过记忆整理。")
    elif pending == 0:
        lines.append("当前会话已经整理到最新的用户消息。")
    else:
        lines.append(f"上次整理到 {pending} 条用户消息之前。")
    if last_content:
        lines.extend(
            ["", "最后已整理的用户消息：", f"“{_preview(last_content)}”"]
        )
    lines.extend(
        [
            "",
            f"尚未整理的用户消息数：{pending}",
            f"当前会话消息数：{total_messages}",
        ]
    )
    return "\n".join(lines)


def _cache_status(
    db_path: Path | None,
    session_key: str,
    content: str,
) -> str:
    if db_path is None or not db_path.exists():
        return "暂无 KVCache 数据（observe 数据库不存在）。"
    parts = content.strip().split()
    limit = 5
    if len(parts) > 1:
        try:
            limit = max(1, min(30, int(parts[1])))
        except ValueError:
            pass
    try:
        connection = sqlite3.connect(
            db_path.resolve().as_uri() + "?mode=ro",
            uri=True,
        )
        try:
            rows = connection.execute(
                """
                SELECT assistant_response, ts, react_cache_prompt_tokens,
                       react_cache_hit_tokens
                FROM turns
                WHERE session_key = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_key, limit),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc) or "no such column" in str(exc):
            return "暂无 KVCache 数据（observe 数据表或字段不存在）。"
        return "KVCache 查询失败。"
    if not rows:
        return "暂无 KVCache 数据。"

    prompt_total = sum(int(row[2] or 0) for row in rows)
    hit_total = sum(int(row[3] or 0) for row in rows)
    overall_pct = hit_total / prompt_total * 100 if prompt_total else 0.0
    lines = [
        f"⚡ KVCache · 最近 {len(rows)} 轮",
        "",
        f"命中率  {overall_pct:.1f}%  {_pct_bar(overall_pct)}",
        f"Token  {hit_total:,} / {prompt_total:,}",
    ]
    for assistant_response, ts, prompt_tokens, hit_tokens in rows:
        prompt = int(prompt_tokens or 0)
        hit = int(hit_tokens or 0)
        pct = hit / prompt * 100 if prompt else 0.0
        lines.extend(
            [
                "",
                "",
                f"{_format_ts(str(ts))}   {_pct_emoji(pct)} "
                f"{pct:.1f}%  {_pct_bar(pct)}",
                f"    {hit:,} / {prompt:,} tokens",
            ]
        )
        preview = _preview(str(assistant_response or ""), limit=72)
        if preview:
            lines.append(f"    {preview}")
    return "\n".join(lines)


def _format_ts(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(_BEIJING_TZ)
        return f"{parsed.month}-{parsed.day} {parsed.hour:02d}:{parsed.minute:02d}"
    except ValueError:
        pass
    return value


def _preview(text: str, limit: int = 80) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _pct_bar(pct: float, width: int = 10) -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _pct_emoji(pct: float) -> str:
    if pct >= 80:
        return "🟢"
    if pct >= 40:
        return "🟡"
    return "🔴"
