"""Transactional Outbox 发布与 Redis 丢失恢复。"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta

from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.redis_queue import PublishedJob, RedisTaskQueue

Failpoint = Callable[[str], None]


class OutboxDispatcher:
    def __init__(
        self,
        repository: OperationalRepository,
        queue: RedisTaskQueue,
        *,
        owner_id: str,
        claim_ttl: timedelta = timedelta(seconds=30),
        failpoint: Failpoint | None = None,
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.owner_id = owner_id
        self.claim_ttl = claim_ttl
        self.failpoint = failpoint

    async def dispatch_one(self, *, now: datetime) -> bool:
        event = self.repository.claim_next_outbox(
            owner_id=self.owner_id,
            now=now,
            claim_ttl=self.claim_ttl,
        )
        if event is None:
            return False
        payload = json.loads(event.payload_json)
        if self.failpoint is not None:
            self.failpoint("after_commit_before_publish")
        try:
            await self.queue.publish(
                PublishedJob(
                    job_id=str(payload["job_id"]),
                    kind=str(payload["kind"]),
                    priority=int(payload["priority"]),
                    session_key=str(payload["session_key"]),
                    payload_json=event.payload_json,
                )
            )
        except Exception as exc:
            self.repository.mark_outbox_retry(
                event.outbox_id,
                owner_id=self.owner_id,
                now=now,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        if self.failpoint is not None:
            self.failpoint("after_publish_before_mark")
        if not self.repository.mark_outbox_published(
            event.outbox_id,
            owner_id=self.owner_id,
            now=now,
        ):
            raise RuntimeError("Outbox claim 已失效，无法标记 published")
        return True


class QueueReconciler:
    def __init__(self, repository: OperationalRepository, queue: RedisTaskQueue) -> None:
        self.repository = repository
        self.queue = queue

    async def reconcile(self, *, now: datetime) -> tuple[str, ...]:
        queued = self.repository.queued_job_ids()
        missing = await self.queue.missing_mirrors(queued)
        return self.repository.requeue_published_outboxes(missing, now=now)


__all__ = ["OutboxDispatcher", "QueueReconciler"]
