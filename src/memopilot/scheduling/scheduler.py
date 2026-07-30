"""将系统生成的任务发布到 Redis。"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.producer import TaskProducer

logger = logging.getLogger(__name__)


class ScheduledTurnPipeline:
    """按 Tick 发布内存、定时和主动任务。"""

    def __init__(
        self,
        scheduler: TaskProducer,
        queue: object,
        *,
        max_publish_per_tick: int = 100,
    ) -> None:
        if max_publish_per_tick <= 0:
            raise ValueError("max_publish_per_tick 必须大于 0")
        self.scheduler = scheduler
        self.queue = queue
        self.max_publish_per_tick = max_publish_per_tick

    async def run_once(self, *, now: datetime | None = None) -> int:
        current = now or self.scheduler.clock()
        result = self.scheduler.tick(now=current)
        published = 0
        for task in (result.memory, *result.schedules.tasks, result.proactive):
            if isinstance(task, AgentTask):
                message_id = await self.queue.publish_task_once(task)  # type: ignore[attr-defined]
                published += message_id is not None
        return published

    async def run_forever(self) -> None:
        logger.info("ScheduledTurnPipeline started")
        try:
            while True:
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Scheduler 任务发布失败")
                await self.scheduler.sleep(self.scheduler.poll_interval_seconds)
        finally:
            logger.info("ScheduledTurnPipeline stopped")


__all__ = ["ScheduledTurnPipeline"]
