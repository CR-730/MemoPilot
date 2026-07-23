"""从最新原型迁移的定时表达式解析与触发时间计算。"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

_DURATION_RE = re.compile(r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def parse_duration(value: str) -> timedelta:
    """解析 ``30s``、``5m``、``2h``、``1h30m`` 等时长。"""
    normalized = value.strip()
    match = _DURATION_RE.match(normalized)
    if not match or not any(match.groups()):
        raise ValueError(
            f"无效的时间间隔: {normalized!r}，示例: '30s', '5m', '2h', '1h30m'"
        )
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def parse_when_at(
    value: str,
    timezone_name: str = "UTC",
    now_fn: Callable[[], datetime] | None = None,
) -> datetime:
    """解析 ``HH:MM`` 或 ISO datetime；已过的钟点顺延到次日。"""
    timezone = ZoneInfo(timezone_name)
    current = (now_fn or (lambda: datetime.now(timezone)))()
    normalized = value.strip()
    if re.match(r"^\d{1,2}:\d{2}$", normalized):
        parsed_time = datetime.strptime(normalized, "%H:%M").time()
        local_now = current.astimezone(timezone)
        result = local_now.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=0,
            microsecond=0,
        )
        if result <= local_now:
            result += timedelta(days=1)
        return result
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"无法解析时间: {normalized!r}，示例: '14:30', '2025-06-01T09:00'"
        ) from exc
    return result.replace(tzinfo=timezone) if result.tzinfo is None else result


def is_cron_expr(value: str) -> bool:
    return len(value.strip().split()) in (5, 6)


def next_cron_fire(expression: str, timezone_name: str, after: datetime) -> datetime:
    """计算五字段或带秒六字段 cron 的下一次 UTC 触发时间。"""
    parts = expression.strip().split()
    timezone = ZoneInfo(timezone_name)
    if len(parts) == 5:
        trigger = CronTrigger.from_crontab(expression, timezone=timezone)
    elif len(parts) == 6:
        second, minute, hour, day, month, day_of_week = parts
        trigger = CronTrigger(
            second=second,
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone=timezone,
        )
    else:
        raise ValueError(f"无效的 cron 表达式: {expression!r}")
    result = trigger.get_next_fire_time(None, after.astimezone(timezone))
    if result is None:
        raise ValueError(f"无效的 cron 表达式: {expression!r}")
    normalized: datetime = result.astimezone(UTC)
    return normalized


def compute_fire_at(
    schedule_kind: str,
    when: str,
    *,
    timezone_name: str = "UTC",
    received_at: datetime | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> datetime:
    """计算首次名义触发时间；``after`` 以可信入站时间为基准。"""
    timezone = ZoneInfo(timezone_name)
    current = (now_fn or (lambda: datetime.now(timezone)))()
    if schedule_kind == "at":
        result = parse_when_at(when, timezone_name, now_fn)
    elif schedule_kind == "after":
        result = (received_at or current) + parse_duration(when)
    elif schedule_kind == "every":
        result = (
            next_cron_fire(when, timezone_name, current)
            if is_cron_expr(when)
            else current + parse_duration(when)
        )
    else:
        raise ValueError(f"未知触发类型: {schedule_kind!r}，须为 at/after/every")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone)
    return result.astimezone(UTC)


def advance_every(
    expression: str,
    *,
    timezone_name: str,
    previous_fire_at: datetime,
    after: datetime,
) -> datetime:
    """Coalesce 错过的周期，只返回严格晚于 ``after`` 的下一次边界。"""
    if is_cron_expr(expression):
        return next_cron_fire(expression, timezone_name, after)
    interval = parse_duration(expression)
    if interval <= timedelta(0):
        raise ValueError("周期必须大于 0")
    if previous_fire_at > after:
        return previous_fire_at.astimezone(UTC)
    elapsed = after - previous_fire_at
    skipped_windows = elapsed // interval + 1
    return (previous_fire_at + interval * skipped_windows).astimezone(UTC)


__all__ = [
    "advance_every",
    "compute_fire_at",
    "is_cron_expr",
    "next_cron_fire",
    "parse_duration",
    "parse_when_at",
]
