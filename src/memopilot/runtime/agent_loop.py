"""单进程 AgentLoop：按优先级重放 Pending 后消费新消息。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from memopilot.persistence.conversation import StaleActivityError
from memopilot.runtime.task_dispatcher import TaskDispatcher
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.redis_queue import QueueMessage, RedisTaskQueue

logger = logging.getLogger(__name__)


class AgentLoop:
    """队列读取、派生任务发布和 ACK 的唯一所有者。"""

    consumer_id = "memopilot-agent"

    def __init__(
        self,
        queue: RedisTaskQueue,
        dispatcher: TaskDispatcher,
        *,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self._queue = queue
        self._dispatcher = dispatcher
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._running = False
        self._current: asyncio.Task[None] | None = None
        self._current_message: QueueMessage | None = None
        self._stop_requested = False
        self._preempted = False

    async def run_once(self) -> bool:
        message = await self._queue.read_next(consumer_id=self.consumer_id)
        if message is None:
            return False
        self._current_message = message
        self._current = asyncio.create_task(self._run_message(message))
        try:
            await self._wait_for_current(message)
        except asyncio.CancelledError:
            if self._stop_requested and message.priority == 0:
                await self._queue.acknowledge(message)
                return True
            if self._preempted:
                if message.priority == 2:
                    await self._queue.acknowledge(message)
                return True
            raise
        finally:
            self._current = None
            self._current_message = None
            self._stop_requested = False
            self._preempted = False
        return True

    async def _wait_for_current(self, message: QueueMessage) -> None:
        assert self._current is not None
        await self._current

    async def _run_message(self, message: QueueMessage) -> None:
        payload = json.loads(message.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("任务 payload 必须是 JSON 对象")
        logger.info(
            "AgentLoop 处理消息 kind=%s session=%s preview=%r",
            message.kind,
            message.session_key,
            str(payload.get("content", ""))[:80],
        )
        try:
            tasks = await self._dispatcher.dispatch(
                AgentTask(
                    message.task_id,
                    message.kind,
                    message.priority,
                    message.session_key,
                    {str(key): value for key, value in payload.items()},
                    self._clock(),
                ),
                now=self._clock(),
            )
        except StaleActivityError:
            if message.priority != 2:
                raise
            await self._queue.acknowledge(message)
            return
        for task in tasks:
            await self._queue.publish_task_once(task)
        await self._queue.acknowledge(message)

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
                    logger.exception("Agent 任务处理失败，保留 Pending 等待重放")
                    processed = False
                if not processed:
                    await self._sleep(idle_interval)
        finally:
            logger.info("AgentLoop 停止")

    def stop(self) -> None:
        self._running = False

    def cancel_current(self) -> bool:
        """`/stop` 只取消当前 P0；其他取消保持 Pending。"""
        if self._current is None or self._current.done() or self._current_message is None:
            return False
        if self._current_message.priority != 0:
            return False
        self._stop_requested = True
        self._current.cancel()
        return True

    def preempt_for_p0(self) -> bool:
        """新 P0 入队后立即抢占当前低优先级任务。"""
        if self._current is None or self._current.done() or self._current_message is None:
            return False
        if self._current_message.priority == 0:
            return False
        self._preempted = True
        self._current.cancel()
        return True


__all__ = ["AgentLoop"]
