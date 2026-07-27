"""定时任务用例服务。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast
from zoneinfo import ZoneInfo

from memopilot.scheduling.contracts import (
    CreateSchedule,
    DueScanResult,
    ExecutionMode,
    ScheduledTask,
    ScheduleKind,
)
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.scheduling.time_rules import compute_fire_at


class ScheduleService:
    def __init__(self, repository: ScheduleRepository) -> None:
        self.repository = repository

    def schedule(
        self,
        *,
        session_key: str,
        received_at: datetime,
        schedule_kind: str,
        when: str,
        execution_mode: str,
        message: str | None = None,
        prompt: str | None = None,
        name: str | None = None,
        timezone: str = "Asia/Shanghai",
        now: datetime | None = None,
    ) -> ScheduledTask:
        if schedule_kind not in {"at", "after", "every"}:
            raise ValueError("schedule_kind 必须为 at、after 或 every")
        if execution_mode not in {"instant", "agent"}:
            raise ValueError("execution_mode 必须为 instant 或 agent")
        if execution_mode == "instant" and not (message or "").strip():
            raise ValueError("instant 模式必须提供 message")
        if execution_mode == "agent" and not (prompt or "").strip():
            raise ValueError("agent 模式必须提供 prompt")
        received_at = (
            received_at.replace(tzinfo=ZoneInfo(timezone))
            if received_at.tzinfo is None
            else received_at
        )
        current = now or datetime.now(UTC)
        next_run_at = compute_fire_at(
            schedule_kind,
            when,
            timezone_name=timezone,
            received_at=received_at,
            now_fn=lambda: current,
        )
        payload: Mapping[str, object] = (
            {"message": message} if execution_mode == "instant" else {"prompt": prompt}
        )
        return self.repository.create(
            CreateSchedule(
                session_key=session_key,
                schedule_kind=cast(ScheduleKind, schedule_kind),
                schedule_expression=when,
                execution_mode=cast(ExecutionMode, execution_mode),
                payload=payload,
                next_run_at=next_run_at,
                timezone=timezone,
                name=name,
                created_at=received_at,
            )
        )

    def list(self, session_key: str) -> tuple[ScheduledTask, ...]:
        return self.repository.list_for_session(session_key)

    def cancel(
        self,
        session_key: str,
        *,
        task_id: str | None = None,
        name: str | None = None,
    ) -> tuple[str, ...]:
        return self.repository.cancel(session_key, task_id=task_id, name=name)

    def scan_due(self, *, now: datetime) -> DueScanResult:
        return self.repository.enqueue_due(now=now)


__all__ = ["ScheduleService"]
