"""Scheduler 进程中的记忆维护任务生产器。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from memopilot.tasks.operational import EnqueueResult, OperationalRepository

logger = logging.getLogger(__name__)


class MemoryMaintenanceScheduler:
    def __init__(
        self,
        repository: OperationalRepository,
        *,
        enabled: bool,
        interval: timedelta,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if interval.total_seconds() <= 0:
            raise ValueError("optimizer interval 必须大于 0")
        self.repository = repository
        self.enabled = enabled
        self.interval = interval
        self.clock = clock or (lambda: datetime.now(UTC))

    def tick(self, *, now: datetime | None = None) -> EnqueueResult | None:
        current = now or self.clock()
        self.repository.requeue_failed_memory_jobs(now=current)
        if not self.enabled:
            return None
        bucket = int(current.timestamp() // self.interval.total_seconds())
        return self.repository.enqueue_memory_optimizer(bucket=bucket, now=current)

    async def run_forever(self) -> None:
        while True:
            try:
                self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("记忆维护 Tick 创建失败，下个周期继续重试")
            await asyncio.sleep(min(60.0, self.interval.total_seconds()))


__all__ = ["MemoryMaintenanceScheduler"]
