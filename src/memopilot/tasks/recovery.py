"""对 Redis Pending 消息执行带会话前置条件的精确接管。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.redis_queue import QueueMessage, RedisTaskQueue


class PendingDisposition(StrEnum):
    RESUME = "resume"
    CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class PendingClaim:
    message: QueueMessage
    disposition: PendingDisposition


class PendingMessageReclaimer:
    """先检查 lease 和 SQLite 心跳，再用 XCLAIM 认领具体消息。"""

    def __init__(
        self,
        repository: OperationalRepository,
        queue: RedisTaskQueue,
        leases: SessionLeaseManager,
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.leases = leases

    async def reclaim_one(
        self,
        *,
        priority: int,
        consumer_id: str,
        min_idle: timedelta,
        heartbeat_before: datetime,
    ) -> PendingClaim | None:
        min_idle_ms = int(min_idle.total_seconds() * 1000)
        if min_idle_ms < 1:
            raise ValueError("Pending 最小空闲时间必须至少为 1ms")
        after_message_id: str | None = None
        while True:
            candidates = await self.queue.pending_entries(
                priority=priority,
                min_idle_ms=min_idle_ms,
                after_message_id=after_message_id,
            )
            if not candidates:
                return None
            for candidate in candidates:
                message = await self.queue.load_message(
                    priority=priority,
                    message_id=candidate.message_id,
                )
                if message is None:
                    continue
                raw_disposition = self.repository.pending_recovery_disposition(
                    message.job_id,
                    heartbeat_before=heartbeat_before,
                )
                if raw_disposition is None:
                    continue
                disposition = PendingDisposition(raw_disposition)
                if disposition is PendingDisposition.RESUME and not await self.leases.is_absent(
                    message.session_key
                ):
                    continue
                claimed = await self.queue.claim_pending(
                    message,
                    consumer_id=consumer_id,
                    min_idle_ms=min_idle_ms,
                )
                if claimed is not None:
                    return PendingClaim(claimed, disposition)
            after_message_id = candidates[-1].message_id


__all__ = ["PendingClaim", "PendingDisposition", "PendingMessageReclaimer"]
