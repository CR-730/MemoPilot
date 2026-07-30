"""系统 Tick 任务生成。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from memopilot.persistence.conversation import ConversationRepository
from memopilot.scheduling.contracts import DueScanResult
from memopilot.tasks.agent_task import AgentTask

logger = logging.getLogger(__name__)


class UserScheduleService(Protocol):
    def scan_due(self, *, now: datetime) -> DueScanResult: ...


@dataclass(frozen=True, slots=True)
class SystemTickResult:
    memory: object
    schedules: DueScanResult
    proactive: AgentTask | None


class TaskProducer:
    """生成内存、定时和主动任务。"""

    def __init__(
        self,
        repository: ConversationRepository,
        *,
        schedule_service: UserScheduleService,
        memory_optimizer_enabled: bool,
        memory_optimizer_interval: timedelta,
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
        self.schedule_service = schedule_service
        self.memory_optimizer_enabled = memory_optimizer_enabled
        self.memory_optimizer_interval = memory_optimizer_interval
        self.proactive_tick_seconds = proactive_tick_seconds
        self.proactive_enabled = proactive_enabled
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds

    def tick(self, *, now: datetime | None = None) -> SystemTickResult:
        current = now or self.clock()
        memory = self._memory_task(current)
        schedules = self.schedule_service.scan_due(now=current)
        target = self.repository.get_single_private_session() if self.proactive_enabled else None
        if target is None:
            return SystemTickResult(memory, schedules, None)
        bucket = int(current.timestamp() // self.proactive_tick_seconds)
        return SystemTickResult(
            memory,
            schedules,
            AgentTask(
                task_id=f"proactive.tick:{target.session_key}:{bucket}",
                kind="proactive.tick",
                priority=2,
                session_key=target.session_key,
                payload={"bucket": bucket, "channel": target.channel, "chat_id": target.chat_id,
                         "activity_version": target.activity_version},
                created_at=current,
            ),
        )

    def _memory_task(self, now: datetime) -> AgentTask | None:
        if not self.memory_optimizer_enabled:
            return None
        self.repository.ensure_system_session("system:memory", chat_id="memory", now=now)
        bucket = int(now.timestamp() // self.memory_optimizer_interval.total_seconds())
        return AgentTask(f"memory.optimize:{bucket}", "memory.optimize", 3, "system:memory",
                         {"bucket": bucket}, now)

    async def run_forever(self) -> None:
        while True:
            try:
                self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler Tick 失败")
            await self.sleep(self.poll_interval_seconds)


__all__ = ["SystemTickResult", "TaskProducer", "UserScheduleService"]
