"""定时任务的领域合同。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ScheduleKind = Literal["at", "after", "every"]
ExecutionMode = Literal["instant", "agent"]


@dataclass(frozen=True, slots=True)
class CreateSchedule:
    session_key: str
    schedule_kind: ScheduleKind
    schedule_expression: str
    execution_mode: ExecutionMode
    payload: Mapping[str, object]
    next_run_at: datetime
    created_at: datetime
    timezone: str = "Asia/Shanghai"
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduledTask:
    task_id: str
    session_key: str
    schedule_kind: ScheduleKind
    schedule_expression: str
    execution_mode: ExecutionMode
    payload: Mapping[str, object]
    next_run_at: datetime | None
    timezone: str
    name: str | None
    enabled: bool
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ScheduledExecution:
    execution_id: str
    task_id: str
    scheduled_at: datetime
    state: str
    job_id: str | None


@dataclass(frozen=True, slots=True)
class DueScanResult:
    queued: tuple[ScheduledExecution, ...] = ()
    missed: tuple[ScheduledExecution, ...] = ()


__all__ = [
    "CreateSchedule",
    "DueScanResult",
    "ExecutionMode",
    "ScheduleKind",
    "ScheduledExecution",
    "ScheduledTask",
]
