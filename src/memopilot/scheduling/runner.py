"""Scheduler 进程的固定周期任务生产器。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from memopilot.scheduling.contracts import DueScanResult
from memopilot.tasks.operational import EnqueueResult, OperationalRepository

logger = logging.getLogger(__name__)


class MemoryScheduler(Protocol):
    def tick(self, *, now: datetime) -> object: ...


class UserScheduleService(Protocol):
    def scan_due(self, *, now: datetime) -> DueScanResult: ...


class OutboxPublisher(Protocol):
    async def dispatch_one(self, *, now: datetime) -> bool: ...


@dataclass(frozen=True, slots=True)
class SystemTickResult:
    memory: object
    schedules: DueScanResult
    proactive: EnqueueResult | None


class SystemScheduler:
    """组合三类生产器；只提交 SQLite 事实与 Outbox。"""

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
        proactive = self.repository.enqueue_proactive_if_session_idle(
            session_key=target.session_key,
            idempotency_key=f"proactive.tick:{target.session_key}:{bucket}",
            activity_version=target.activity_version,
            payload={"bucket": bucket, "chat_id": target.chat_id},
            now=current,
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


class SchedulerProcess:
    """Scheduler 进程入口：生成 SQLite 事实并把 Outbox 镜像发布到 Redis。"""

    def __init__(
        self,
        scheduler: SystemScheduler,
        outbox: OutboxPublisher,
        *,
        max_publish_per_tick: int = 100,
    ) -> None:
        if max_publish_per_tick <= 0:
            raise ValueError("max_publish_per_tick 必须大于 0")
        self.scheduler = scheduler
        self.outbox = outbox
        self.max_publish_per_tick = max_publish_per_tick

    async def run_once(self, *, now: datetime | None = None) -> int:
        current = now or self.scheduler.clock()
        self.scheduler.tick(now=current)
        published = 0
        for _ in range(self.max_publish_per_tick):
            if not await self.outbox.dispatch_one(now=current):
                break
            published += 1
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


__all__ = ["SchedulerProcess", "SystemScheduler", "SystemTickResult"]
