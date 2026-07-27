"""Redis 后台任务循环。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol

from memopilot.tasks.lease import SessionLease, SessionLeaseManager
from memopilot.tasks.operational import LostLeaseError, StaleActivityError
from memopilot.tasks.redis_queue import PublishedTask, QueueMessage, RedisTaskQueue
from memopilot.tasks.session_coordination import RedisSessionCoordinator

logger = logging.getLogger(__name__)


class BackgroundTaskExecutor(Protocol):
    async def execute(
        self,
        message: QueueMessage,
        *,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> str: ...


class BackgroundTaskLoop:
    """按优先级消费 Redis 后台任务，并用 Lease 防止会话并发执行。"""

    def __init__(
        self,
        queue: RedisTaskQueue,
        leases: SessionLeaseManager,
        executor: BackgroundTaskExecutor,
        *,
        owner_id: str,
        clock: Callable[[], datetime] | None = None,
        heartbeat_interval: float | None = None,
        pending_min_idle: timedelta = timedelta(seconds=60),
        pending_scan_interval: float = 5.0,
        interrupt_poll_interval: float = 0.1,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
        session_coordinator: RedisSessionCoordinator | None = None,
    ) -> None:
        if pending_min_idle.total_seconds() <= 0:
            raise ValueError("Pending 空闲阈值必须大于 0")
        if interrupt_poll_interval <= 0:
            raise ValueError("interrupt_poll_interval 必须大于 0")
        if pending_scan_interval <= 0:
            raise ValueError("pending_scan_interval 必须大于 0")
        self._queue = queue
        self._leases = leases
        self._executor = executor
        self._owner_id = owner_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._pending_min_idle = pending_min_idle
        self._pending_scan_interval = pending_scan_interval
        self._next_pending_scan_at = float("-inf")
        self._heartbeat_interval = heartbeat_interval or max(0.05, leases.ttl_ms / 3000)
        if self._heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval 必须大于 0")
        self._interrupt_poll_interval = interrupt_poll_interval
        self._monotonic = monotonic
        self._sleep = sleep
        self._session_coordinator = session_coordinator

    async def run_once(self) -> bool:
        message = await self._queue.read_next(consumer_id=self._owner_id)
        if message is None:
            current = self._monotonic()
            if current < self._next_pending_scan_at:
                return False
            self._next_pending_scan_at = current + self._pending_scan_interval
            message = await self._reclaim_pending()
            if message is None:
                return False

        if await self._user_turn_active(message.session_key):
            await self._defer_after_user_activity(message)
            return True

        lease = await self._leases.acquire(
            message.session_key,
            owner_id=self._owner_id,
            now=self._clock(),
        )
        if lease is None:
            await self._requeue(message)
            return True

        try:
            payload = json.loads(message.payload_json)
            if not isinstance(payload, dict):
                raise ValueError("后台任务 payload 必须是 JSON 对象")
            try:
                preempted = await self._execute_with_heartbeat(
                    message,
                    {str(key): value for key, value in payload.items()},
                    lease,
                )
            except StaleActivityError:
                if message.kind != "proactive.tick":
                    raise
                await self._queue.acknowledge(message)
                return True
            if preempted:
                await self._defer_after_user_activity(message)
            else:
                await self._queue.acknowledge(message)
            return True
        finally:
            await self._leases.release(lease)

    async def run_forever(self, *, idle_interval: float = 0.25) -> None:
        if idle_interval <= 0:
            raise ValueError("idle_interval 必须大于 0")
        await self._queue.ensure_consumer_groups()
        while True:
            try:
                processed = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("后台任务处理失败，继续消费后续任务")
                processed = False
            if not processed:
                await self._sleep(idle_interval)

    async def _reclaim_pending(self) -> QueueMessage | None:
        min_idle_ms = int(self._pending_min_idle.total_seconds() * 1000)
        for priority in range(4):
            pending_entries = await self._queue.pending_entries(
                priority=priority,
                min_idle_ms=min_idle_ms,
            )
            for pending in pending_entries:
                message = await self._queue.load_message(
                    priority=priority,
                    message_id=pending.message_id,
                )
                if message is None or not await self._leases.is_absent(message.session_key):
                    continue
                claimed = await self._queue.claim_pending(
                    message,
                    consumer_id=self._owner_id,
                    min_idle_ms=min_idle_ms,
                )
                if claimed is not None:
                    return claimed
        return None

    async def _requeue(self, message: QueueMessage) -> None:
        await self._queue.acknowledge_requeued(message)
        await self._queue.publish(
            PublishedTask(
                message.task_id,
                message.kind,
                message.priority,
                message.session_key,
                message.payload_json,
            )
        )

    async def _defer_after_user_activity(self, message: QueueMessage) -> None:
        if message.kind == "proactive.tick":
            await self._queue.acknowledge(message)
            return
        await self._requeue(message)

    async def _execute_with_heartbeat(
        self,
        message: QueueMessage,
        payload: dict[str, object],
        lease: SessionLease,
    ) -> bool:
        execution = asyncio.create_task(
            self._executor.execute(
                message,
                payload=payload,
                lease=lease,
                now=self._clock(),
            )
        )
        next_heartbeat = self._monotonic() + self._heartbeat_interval
        try:
            while True:
                timeout = min(
                    self._interrupt_poll_interval,
                    max(0.0, next_heartbeat - self._monotonic()),
                )
                done, _ = await asyncio.wait({execution}, timeout=timeout)
                if done:
                    await execution
                    return False
                if await self._background_stop_requested(message.session_key):
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    return True
                if self._monotonic() >= next_heartbeat:
                    if not await self._leases.renew(lease, now=self._clock()):
                        execution.cancel()
                        await asyncio.gather(execution, return_exceptions=True)
                        raise LostLeaseError("会话 Lease 续租失败，后台任务已停止提交")
                    next_heartbeat = self._monotonic() + self._heartbeat_interval
        except BaseException:
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise

    async def _user_turn_active(self, session_key: str) -> bool:
        return (
            self._session_coordinator is not None
            and await self._session_coordinator.user_turn_active(session_key)
        )

    async def _background_stop_requested(self, session_key: str) -> bool:
        return (
            self._session_coordinator is not None
            and await self._session_coordinator.background_stop_requested(session_key)
        )


__all__ = ["BackgroundTaskExecutor", "BackgroundTaskLoop"]
