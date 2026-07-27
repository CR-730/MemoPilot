"""固定周期任务生产与 Redis 发布服务。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from memopilot.scheduling.contracts import DueScanResult
from memopilot.tasks.background import BackgroundTask
from memopilot.tasks.operational import OperationalRepository

logger = logging.getLogger(__name__)


class MemoryScheduler(Protocol):
    def tick(self, *, now: datetime) -> object: ...


class UserScheduleService(Protocol):
    def scan_due(self, *, now: datetime) -> DueScanResult: ...


class SessionCoordinator(Protocol):
    async def session_busy(self, session_key: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class SystemTickResult:
    memory: object
    schedules: DueScanResult
    proactive: BackgroundTask | None


class SystemScheduler:
    """组合记忆、定时任务和主动唤醒三类生产器。"""

    def __init__(
        self,
        repository: OperationalRepository,
        *,
        memory_scheduler: MemoryScheduler,
        schedule_service: UserScheduleService,
        proactive_tick_seconds: int,
        proactive_enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        if proactive_tick_seconds <= 0:
            raise ValueError("proactive_tick_seconds 必须大于 0")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds 必须大于 0")
        self.repository = repository
        self.memory_scheduler = memory_scheduler
        self.schedule_service = schedule_service
        self.proactive_tick_seconds = proactive_tick_seconds
        self.proactive_enabled = proactive_enabled
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds

    def tick(self, *, now: datetime | None = None) -> SystemTickResult:
        current = now or self.clock()
        memory = self.memory_scheduler.tick(now=current)
        schedules = self.schedule_service.scan_due(now=current)
        target = self.repository.get_single_private_session() if self.proactive_enabled else None
        if target is None:
            return SystemTickResult(memory, schedules, None)
        bucket = int(current.timestamp() // self.proactive_tick_seconds)
        proactive = BackgroundTask(
            task_id=f"proactive.tick:{target.session_key}:{bucket}",
            kind="proactive.tick",
            priority=2,
            session_key=target.session_key,
            payload={
                "bucket": bucket,
                "channel": target.channel,
                "chat_id": target.chat_id,
                "activity_version": target.activity_version,
            },
            created_at=current,
        )
        return SystemTickResult(memory, schedules, proactive)

    async def run_forever(self) -> None:
        while True:
            try:
                self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("系统 Scheduler Tick 失败，下次轮询继续")
            await self.sleep(self.poll_interval_seconds)


class SchedulerService:
    """生成到期任务并发布到 Redis。"""

    def __init__(
        self,
        scheduler: SystemScheduler,
        queue: object,
        *,
        max_publish_per_tick: int = 100,
        session_coordinator: SessionCoordinator | None = None,
    ) -> None:
        if max_publish_per_tick <= 0:
            raise ValueError("max_publish_per_tick 必须大于 0")
        self.scheduler = scheduler
        self.queue = queue
        self.max_publish_per_tick = max_publish_per_tick
        self.session_coordinator = session_coordinator

    async def run_once(self, *, now: datetime | None = None) -> int:
        current = now or self.scheduler.clock()
        result = self.scheduler.tick(now=current)
        published = 0
        for task in (result.memory, *result.schedules.tasks, result.proactive):
            if isinstance(task, BackgroundTask):
                if (
                    task.kind == "proactive.tick"
                    and self.session_coordinator is not None
                    and await self.session_coordinator.session_busy(task.session_key)
                ):
                    continue
                message_id = await self.queue.publish_task_once(task)  # type: ignore[attr-defined]
                published += message_id is not None
        return published

    async def run_forever(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler 周期失败，下次轮询继续")
            await self.scheduler.sleep(self.scheduler.poll_interval_seconds)


__all__ = ["SchedulerService", "SystemScheduler", "SystemTickResult"]
