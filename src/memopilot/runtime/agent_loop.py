"""按原型 MessageBus 驱动被动 Agent Turn。"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.bus.events import InboundMessage, OutboundMessage
from memopilot.bus.queue import MessageBus
from memopilot.channels.contracts import InterruptAcknowledgement
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.react import ReActProgressObserver
from memopilot.tasks.background import BackgroundTask
from memopilot.tasks.session_coordination import RedisSessionCoordinator

logger = logging.getLogger(__name__)


class HistoryRecord(Protocol):
    role: str
    content: str


class PassiveTurnStore(Protocol):
    def list_recent_messages(self, session_key: str, *, limit: int) -> Sequence[HistoryRecord]: ...

    def record_inbound_activity(self, message: InboundMessage) -> int: ...

    def commit_turn(
        self,
        message: InboundMessage,
        *,
        assistant_content: str,
        cited_memory_ids: tuple[str, ...],
        explicitly_memorized_ids: tuple[str, ...],
    ) -> Sequence[BackgroundTask] | Awaitable[Sequence[BackgroundTask] | None] | None: ...


class BackgroundTaskPublisher(Protocol):
    async def publish_task_once(self, task: BackgroundTask) -> str | None: ...


class AgentLoop:
    """原型式的被动主循环：Bus → Runtime → Bus。"""

    def __init__(
        self,
        *,
        bus: MessageBus,
        runtime: AgentRuntime,
        store: PassiveTurnStore,
        short_term_message_limit: int = 12,
        clock: Callable[[], datetime] | None = None,
        progress_factory: Callable[[InboundMessage], ReActProgressObserver | None] | None = None,
        session_coordinator: RedisSessionCoordinator | None = None,
        background_publisher: BackgroundTaskPublisher | None = None,
    ) -> None:
        if short_term_message_limit < 1:
            raise ValueError("短期消息窗口必须至少包含 1 条消息")
        self.bus = bus
        self.runtime = runtime
        self.store = store
        self.short_term_message_limit = short_term_message_limit
        self.clock = clock or (lambda: datetime.now(UTC))
        self._running = False
        self._active: dict[str, asyncio.Task[None]] = {}
        self._progress_factory = progress_factory
        self._session_coordinator = session_coordinator
        self._background_publisher = background_publisher

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            message = await self.bus.consume_inbound()
            try:
                await self.process(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "被动 Agent Turn 失败 session=%s，主循环继续等待下一条消息",
                    message.session_key,
                )

    def stop(self) -> None:
        self._running = False
        for task in tuple(self._active.values()):
            task.cancel()

    async def request_interrupt(self, message: InboundMessage) -> InterruptAcknowledgement:
        """按原型在内存中取消当前会话的 Turn，不创建额外持久化任务。"""
        task = self._active.get(message.session_key)
        active = task is not None and not task.done()
        if active:
            assert task is not None
            task.cancel()
        return InterruptAcknowledgement(
            message=(
                "已收到停止请求，正在中断当前回复。" if active else "当前没有正在执行的回复。"
            ),
            provider_uuid=str(
                uuid5(
                    NAMESPACE_URL,
                    "memopilot:interrupt:"
                    f"{message.metadata.get('message_id') or message.session_key}",
                )
            ),
        )

    async def process(self, message: InboundMessage) -> None:
        previous = self._active.get(message.session_key)
        if previous is not None and not previous.done():
            await previous
        task = asyncio.create_task(
            self._process_one(message),
            name=f"memopilot-agent-loop:{message.session_key}",
        )
        self._active[message.session_key] = task
        try:
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
        finally:
            if self._active.get(message.session_key) is task:
                self._active.pop(message.session_key, None)

    async def _process_one(self, message: InboundMessage) -> None:
        self.store.record_inbound_activity(message)
        turn_id = str(message.metadata.get("message_id") or message.session_key)
        if self._session_coordinator is not None:
            await self._session_coordinator.request_background_stop(
                message.session_key,
                reason="user_message",
            )
            await self._session_coordinator.begin_user_turn(
                message.session_key,
                turn_id=turn_id,
            )
        history = tuple(
            ChatMessage.user(record.content)
            if record.role == "user"
            else ChatMessage.assistant(content=record.content)
            for record in self.store.list_recent_messages(
                message.session_key,
                limit=self.short_term_message_limit,
            )
            if record.role in {"user", "assistant"}
        )
        progress = self._progress_factory(message) if self._progress_factory else None
        try:
            try:
                result = await self.runtime.run(
                    TurnInput(
                        session_key=message.session_key,
                        content=message.content,
                        history=history,
                        current_user_content=message.content,
                        received_at=message.timestamp,
                    ),
                    progress=progress,
                )
            finally:
                if progress is not None:
                    finalize = getattr(progress, "finalize", None)
                    if callable(finalize):
                        await finalize()
            committed = self.store.commit_turn(
                message,
                assistant_content=result.reply,
                cited_memory_ids=result.cited_memory_ids,
                explicitly_memorized_ids=_explicitly_memorized_ids(result.trace),
            )
            if inspect.isawaitable(committed):
                committed = await committed
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=message.channel,
                    chat_id=message.chat_id,
                    content=result.reply,
                    thinking=result.react.thinking,
                    reply_to=str(message.metadata.get("message_id") or "") or None,
                    metadata={
                        "provider_uuid": str(
                            uuid5(
                                NAMESPACE_URL,
                                "memopilot:reply:"
                                f"{message.metadata.get('message_id') or message.session_key}",
                            )
                        )
                    },
                )
            )
            if committed and self._background_publisher is not None:
                for background_task in committed:
                    await self._background_publisher.publish_task_once(background_task)
        finally:
            if self._session_coordinator is not None:
                await self._session_coordinator.end_user_turn(
                    message.session_key,
                    turn_id=turn_id,
                )
                await self._session_coordinator.clear_background_stop(message.session_key)


def _explicitly_memorized_ids(trace: Sequence[object]) -> tuple[str, ...]:
    item_ids: list[str] = []
    for event in trace:
        if getattr(event, "tool_name", None) != "memorize":
            continue
        if getattr(event, "state", None) != "succeeded":
            continue
        observation = getattr(event, "observation", None)
        result = observation.get("result") if isinstance(observation, dict) else None
        item_id = result.get("item_id") if isinstance(result, dict) else None
        if isinstance(item_id, str) and item_id.strip():
            item_ids.append(item_id.strip())
    return tuple(dict.fromkeys(item_ids))


__all__ = [
    "AgentLoop",
    "BackgroundTaskPublisher",
    "HistoryRecord",
    "PassiveTurnStore",
    "_explicitly_memorized_ids",
]
