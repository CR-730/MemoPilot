"""统一消费 Redis 任务的 AgentLoop。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta

from memopilot.runtime.task_dispatcher import TaskDispatcher
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.lease import SessionLease, SessionLeaseManager
from memopilot.tasks.operational import LostLeaseError, StaleActivityError
from memopilot.tasks.redis_queue import QueueMessage, RedisTaskQueue
from memopilot.tasks.session_coordination import RedisSessionCoordinator

logger = logging.getLogger(__name__)


class AgentLoop:
    """所有 Agent 工作共用优先级、Lease、Fencing、Pending 与 ACK 边界。"""

    def __init__(
        self,
        queue: RedisTaskQueue,
        leases: SessionLeaseManager,
        dispatcher: TaskDispatcher,
        *,
        owner_id: str,
        session_coordinator: RedisSessionCoordinator,
        clock: Callable[[], datetime] | None = None,
        heartbeat_interval: float | None = None,
        pending_min_idle: timedelta = timedelta(seconds=60),
        pending_scan_interval: float = 5.0,
        interrupt_poll_interval: float = 0.1,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        if pending_min_idle.total_seconds() <= 0:
            raise ValueError("Pending 空闲阈值必须大于 0")
        if interrupt_poll_interval <= 0 or pending_scan_interval <= 0:
            raise ValueError("轮询间隔必须大于 0")
        self._queue = queue
        self._leases = leases
        self._dispatcher = dispatcher
        self._owner_id = owner_id
        self._coordinator = session_coordinator
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
        self._running = False

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

        lease = await self._leases.acquire(
            message.session_key,
            owner_id=self._owner_id,
            now=self._clock(),
        )
        if lease is None:
            return True

        try:
            payload = json.loads(message.payload_json)
            if not isinstance(payload, dict):
                raise ValueError("任务 payload 必须是 JSON 对象")
            preview = " ".join(str(payload.get("content") or "").split())[:80]
            logger.info(
                "AgentLoop 处理消息 kind=%s session=%s preview=%r",
                message.kind,
                message.session_key,
                preview,
            )
            if message.kind == "passive.turn":
                await self._coordinator.clear_background_stop(message.session_key)
            try:
                tasks, stop_reason = await self._execute_with_heartbeat(
                    message,
                    {str(key): value for key, value in payload.items()},
                    lease,
                )
            except StaleActivityError:
                if message.kind not in {"proactive.tick", "drift.run"}:
                    raise
                await self._queue.acknowledge(message)
                return True
            if stop_reason is not None:
                if message.kind in {"proactive.tick", "drift.run"} or (
                    message.kind == "passive.turn" and stop_reason == "user_stop"
                ):
                    await self._queue.acknowledge(message)
                if stop_reason == "user_stop":
                    await self._coordinator.clear_background_stop(message.session_key)
                return True
            for task in tasks:
                await self._queue.publish_task_once(task)
            await self._queue.acknowledge(message)
            return True
        finally:
            await self._leases.release(lease)

    async def run_forever(self, *, idle_interval: float = 0.25) -> None:
        if idle_interval <= 0:
            raise ValueError("idle_interval 必须大于 0")
        await self._queue.ensure_consumer_groups()
        self._running = True
        logger.info("AgentLoop 启动")
        try:
            while self._running:
                try:
                    processed = await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Agent 任务处理失败，继续消费后续任务")
                    processed = False
                if not processed:
                    await self._sleep(idle_interval)
        finally:
            logger.info("AgentLoop 停止")

    def stop(self) -> None:
        self._running = False

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

    async def _execute_with_heartbeat(
        self,
        message: QueueMessage,
        payload: dict[str, object],
        lease: SessionLease,
    ) -> tuple[Sequence[AgentTask], str | None]:
        execution = asyncio.create_task(
            self._dispatcher.dispatch(
                AgentTask(
                    message.task_id,
                    message.kind,
                    message.priority,
                    message.session_key,
                    payload,
                    self._clock(),
                ),
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
                    return await execution, None
                stop_reason = await self._coordinator.stop_reason(message.session_key)
                if stop_reason is not None and (
                    message.kind != "passive.turn" or stop_reason == "user_stop"
                ):
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    return (), stop_reason
                if self._monotonic() >= next_heartbeat:
                    if not await self._leases.renew(lease, now=self._clock()):
                        execution.cancel()
                        await asyncio.gather(execution, return_exceptions=True)
                        raise LostLeaseError("会话 Lease 续租失败，任务已停止提交")
                    next_heartbeat = self._monotonic() + self._heartbeat_interval
        except BaseException:
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise


__all__ = ["AgentLoop"]
