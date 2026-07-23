import inspect
from datetime import UTC, datetime, timedelta

import pytest

from memopilot.scheduling.time_rules import (
    advance_every,
    compute_fire_at,
    is_cron_expr,
    next_cron_fire,
    parse_duration,
    parse_when_at,
)


def test_parse_duration_accepts_composite_units() -> None:
    assert parse_duration("1d2h30m4s") == timedelta(days=1, hours=2, minutes=30, seconds=4)


@pytest.mark.parametrize("value", ["", "5", "-1m", "1.5h", "nonsense"])
def test_parse_duration_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="无效的时间间隔"):
        parse_duration(value)


def test_parse_when_at_moves_elapsed_clock_time_to_tomorrow() -> None:
    now = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)

    result = parse_when_at("14:30", "UTC", now_fn=lambda: now)

    assert result == datetime(2026, 7, 22, 14, 30, tzinfo=UTC)


def test_parse_when_at_applies_requested_timezone_to_naive_iso() -> None:
    result = parse_when_at("2026-07-22T09:00:00", "Asia/Shanghai")

    assert result.astimezone(UTC) == datetime(2026, 7, 22, 1, 0, tzinfo=UTC)


def test_after_uses_trusted_received_at_instead_of_current_time() -> None:
    received_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    much_later = datetime(2026, 7, 21, 12, 10, tzinfo=UTC)

    result = compute_fire_at(
        "after",
        "30s",
        timezone_name="UTC",
        received_at=received_at,
        now_fn=lambda: much_later,
    )

    assert result == received_at + timedelta(seconds=30)


def test_every_interval_starts_from_current_time() -> None:
    now = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    assert compute_fire_at("every", "1h", now_fn=lambda: now) == now + timedelta(hours=1)


def test_cron_supports_five_and_six_fields() -> None:
    assert is_cron_expr("0 9 * * *") is True
    assert is_cron_expr("30 0 9 * * *") is True
    after = datetime(2026, 7, 21, 0, 0, tzinfo=UTC)
    assert next_cron_fire("0 9 * * *", "Asia/Shanghai", after) == datetime(
        2026, 7, 21, 1, 0, tzinfo=UTC
    )
    assert next_cron_fire("30 0 9 * * *", "Asia/Shanghai", after) == datetime(
        2026, 7, 21, 1, 0, 30, tzinfo=UTC
    )


def test_interval_coalesce_across_years_uses_constant_time_arithmetic() -> None:
    source = inspect.getsource(advance_every)
    assert "while " not in source

    previous = datetime(2000, 1, 1, tzinfo=UTC)
    after = datetime(2030, 1, 1, tzinfo=UTC)

    assert advance_every(
        "1s", timezone_name="UTC", previous_fire_at=previous, after=after
    ) == after + timedelta(seconds=1)
