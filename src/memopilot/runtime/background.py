"""Redis 任务到五类领域处理器的单次分发。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.bus.events import InboundMessage, TurnCommitted
from memopilot.extensions.events import EventBus
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.history import build_tool_chain, expand_history
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.outbound import DeliveryError, OutboundDispatch, OutboundPort
from memopilot.runtime.react import ReActProgressObserver
from memopilot.tasks.background import BackgroundTask
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.redis_queue import QueueMessage


class ProactiveTaskExecutor(Protocol):
    async def execute_task(
        self,
        *,
        task_id: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> str: ...


class MemoryTaskExecutor(Protocol):
    async def execute(
        self,
        *,
        kind: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
    ) -> None: ...


class DriftTaskExecutor(Protocol):
    async def execute_task(
        self,
        *,
        task_id: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> object: ...


class CoreRunner:
    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        repository: OperationalRepository,
        outbound: OutboundPort,
        memory_tasks: MemoryTaskExecutor,
        proactive: ProactiveTaskExecutor,
        drift: DriftTaskExecutor,
        event_bus: EventBus,
        short_term_message_limit: int = 12,
        progress_factory: Callable[[InboundMessage], ReActProgressObserver | None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.repository = repository
        self.outbound = outbound
        self.memory_tasks = memory_tasks
        self.proactive = proactive
        self.drift = drift
        self.event_bus = event_bus
        self.short_term_message_limit = short_term_message_limit
        self.progress_factory = progress_factory

    async def execute(
        self,
        message: QueueMessage,
        *,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> Sequence[BackgroundTask]:
        if message.kind == "passive.turn":
            return await self._run_passive(message, payload, lease=lease)
        elif message.kind.startswith("memory."):
            await self.memory_tasks.execute(
                kind=message.kind,
                session_key=message.session_key,
                payload=payload,
                lease=lease,
            )
        elif message.kind == "proactive.tick":
            outcome = await self.proactive.execute_task(
                task_id=message.task_id,
                session_key=message.session_key,
                payload=payload,
                lease=lease,
                now=now,
            )
            if outcome == "drift":
                await self.drift.execute_task(
                    task_id=message.task_id,
                    session_key=message.session_key,
                    payload=payload,
                    lease=lease,
                    now=now,
                )
        elif message.kind == "drift.run":
            await self.drift.execute_task(
                task_id=message.task_id,
                session_key=message.session_key,
                payload=payload,
                lease=lease,
                now=now,
            )
        elif message.kind == "schedule.run":
            try:
                await self._run_schedule(message, payload, lease=lease, now=now)
            except BaseException as exc:
                if not isinstance(
                    exc,
                    (asyncio.CancelledError, KeyboardInterrupt, SystemExit, DeliveryError),
                ):
                    execution_id = str(payload.get("execution_id") or "")
                    if execution_id:
                        self.repository.transition_background_schedule(
                            execution_id,
                            lease=lease,
                            outcome="failed",
                            now=now,
                        )
                raise
        else:
            raise ValueError(f"不支持的 Agent 任务: {message.kind}")
        return ()

    async def _run_passive(
        self,
        queue_message: QueueMessage,
        payload: Mapping[str, object],
        *,
        lease: SessionLease,
    ) -> Sequence[BackgroundTask]:
        message = _inbound_message(payload)
        if message.session_key != queue_message.session_key:
            raise ValueError("passive.turn 的 session_key 不一致")
        self.repository.record_inbound_activity(message)
        committed = self.repository.commit_turn(message, lease=lease)
        thinking: str | None = None
        tools_used: list[str] = []
        tool_chain: tuple[dict[str, str], ...] = ()
        cache_prompt_tokens = 0
        cache_hit_tokens = 0
        if committed is None:
            history = expand_history(
                self.repository.list_recent_messages(
                    message.session_key, limit=self.short_term_message_limit
                )
            )
            progress = self.progress_factory(message) if self.progress_factory else None
            try:
                result = await self.runtime.run(
                    TurnInput(
                        session_key=message.session_key,
                        content=message.content,
                        history=history,
                        current_user_content=message.content,
                        media=tuple(message.media),
                        received_at=message.timestamp,
                    ),
                    progress=progress,
                )
            finally:
                if progress is not None:
                    await progress.finalize()  # type: ignore[attr-defined]
            committed = self.repository.commit_turn(
                message,
                assistant_content=result.reply,
                assistant_media=result.media,
                assistant_tool_chain=build_tool_chain(
                    result.react.messages,
                    call_ids=frozenset(item.call.id for item in result.react.tool_chain),
                ),
                cited_memory_ids=result.cited_memory_ids,
                explicitly_memorized_ids=_explicitly_memorized_ids(result.trace),
                lease=lease,
            )
            assert committed is not None
            thinking = result.react.thinking
            tools_used = list(dict.fromkeys(item.call.name for item in result.react.tool_chain))
            tool_chain = tuple(
                {
                    "name": item.call.name,
                    "status": item.observation.status,
                    ("result" if item.observation.ok else "error"): (
                        item.observation.content
                        if item.observation.ok
                        else str(
                            item.observation.error_message
                            or item.observation.error_type
                            or ""
                        )
                    ),
                }
                for item in result.react.tool_chain
            )
            cache_prompt_tokens = result.react.prompt_tokens
            cache_hit_tokens = result.react.prompt_cache_hit_tokens

        if committed.inserted:
            await self.event_bus.fanout(
                TurnCommitted(
                    session_key=message.session_key,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    input_message=message.content,
                    assistant_response=committed.assistant_content,
                    tools_used=tools_used,
                    timestamp=message.timestamp,
                    tool_chain=tool_chain,
                    react_cache_prompt_tokens=cache_prompt_tokens,
                    react_cache_hit_tokens=cache_hit_tokens,
                )
            )
        sent = await self.outbound.dispatch(
            OutboundDispatch(
                channel=message.channel,
                chat_id=message.chat_id,
                content=committed.assistant_content,
                thinking=thinking,
                metadata={
                    "provider_uuid": str(
                        uuid5(
                            NAMESPACE_URL,
                            "memopilot:reply:"
                            f"{message.metadata.get('message_id') or message.session_key}",
                        )
                    )
                },
                media=list(committed.media),
            )
        )
        if not sent:
            raise DeliveryError("被动回复未明确发送成功")
        return committed.background_tasks

    async def _run_schedule(
        self,
        message: QueueMessage,
        payload: Mapping[str, object],
        *,
        lease: SessionLease,
        now: datetime,
    ) -> None:
        execution_id = _required_text(payload, "execution_id")
        current = self.repository.transition_background_schedule(
            execution_id,
            lease=lease,
            outcome="running",
            now=now,
        )
        if current in {"succeeded", "failed", "cancelled"}:
            return
        task_payload = payload.get("payload")
        if not isinstance(task_payload, Mapping):
            raise ValueError("schedule.run payload.payload 必须是对象")
        mode = _required_text(payload, "execution_mode")
        if mode == "instant":
            text = _required_text(task_payload, "message")
        elif mode == "agent":
            result = await self.runtime.run(
                replace(
                    TurnInput(
                        session_key=message.session_key,
                        content=_required_text(task_payload, "prompt"),
                        prompt_scope="scheduled",
                        received_at=now,
                    ),
                    allowed_tool_risks=frozenset({"read-only", "write"}),
                    memory_source_ref=f"task:{message.task_id}",
                ),
            )
            if result.react.infrastructure_error:
                self.repository.transition_background_schedule(
                    execution_id, lease=lease, outcome="failed", now=now
                )
                return
            text = result.reply
        else:
            raise ValueError(f"未知 schedule execution_mode: {mode}")
        if not await self.outbound.dispatch(
            OutboundDispatch(
                channel=_required_text(payload, "channel"),
                chat_id=_required_text(payload, "chat_id"),
                content=text,
            )
        ):
            raise DeliveryError("定时任务结果未明确发送成功")
        self.repository.transition_background_schedule(
            execution_id, lease=lease, outcome="succeeded", now=now
        )


def _inbound_message(payload: Mapping[str, object]) -> InboundMessage:
    timestamp = datetime.fromisoformat(_required_text(payload, "timestamp"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    metadata = payload.get("metadata")
    media = payload.get("media")
    return InboundMessage(
        channel=_required_text(payload, "channel"),
        sender=_required_text(payload, "sender"),
        chat_id=_required_text(payload, "chat_id"),
        content=_required_text(payload, "content"),
        timestamp=timestamp,
        media=[str(item) for item in media] if isinstance(media, list) else [],
        metadata={str(key): value for key, value in metadata.items()}
        if isinstance(metadata, Mapping)
        else {},
    )


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


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Agent 任务缺少 {key}")
    return value


__all__ = [
    "CoreRunner",
    "DriftTaskExecutor",
    "MemoryTaskExecutor",
    "ProactiveTaskExecutor",
    "_explicitly_memorized_ids",
]
