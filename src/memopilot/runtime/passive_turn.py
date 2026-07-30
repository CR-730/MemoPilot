"""Passive reply persistence and delivery pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from memopilot.bus.events import InboundMessage, TurnCommitted
from memopilot.extensions.events import EventBus
from memopilot.runtime.engine import AgentRuntime, SessionHistoryRequest, TurnInput
from memopilot.runtime.history import build_tool_chain
from memopilot.runtime.outbound import DeliveryError, OutboundDispatch, OutboundPort
from memopilot.runtime.react import ReActProgressObserver
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository


class PassiveTurnPipeline:
    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        repository: OperationalRepository,
        outbound: OutboundPort,
        event_bus: EventBus,
        history_limit: int,
        progress_factory: Callable[[InboundMessage], ReActProgressObserver | None] | None = None,
    ) -> None:
        self._runtime, self._repository, self._outbound = runtime, repository, outbound
        self._event_bus, self._history_limit, self._progress_factory = (
            event_bus,
            history_limit,
            progress_factory,
        )

    async def execute_task(
        self, task: AgentTask, *, lease: SessionLease, now: datetime
    ) -> Sequence[AgentTask]:
        del now
        message = _inbound_message(task.payload)
        if message.session_key != task.session_key:
            raise ValueError("passive.turn session_key mismatch")
        self._repository.record_inbound_activity(message)
        committed = self._repository.commit_turn(message, lease=lease)
        thinking: str | None = None
        tools_used: list[str] = []
        tool_chain: tuple[dict[str, str], ...] = ()
        cache_prompt_tokens = 0
        cache_hit_tokens = 0
        if committed is None:
            progress = self._progress_factory(message) if self._progress_factory else None
            try:
                result = await self._runtime.run(
                    TurnInput(
                        message.session_key,
                        message.content,
                        history=SessionHistoryRequest(message.session_key, self._history_limit),
                        current_user_content=message.content,
                        media=tuple(message.media),
                        received_at=message.timestamp,
                    ),
                    progress=progress,
                )
            finally:
                if progress is not None:
                    await progress.finalize()  # type: ignore[attr-defined]
            committed = self._repository.commit_turn(
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
            await self._event_bus.fanout(
                TurnCommitted(
                    message.session_key,
                    message.channel,
                    message.chat_id,
                    message.content,
                    committed.assistant_content,
                    tools_used,
                    message.timestamp,
                    tool_chain,
                    cache_prompt_tokens,
                    cache_hit_tokens,
                )
            )
        provider_uuid = str(
            uuid5(
                NAMESPACE_URL,
                "memopilot:reply:" + str(message.metadata.get("message_id") or message.session_key),
            )
        )
        sent = await self._outbound.dispatch(
            OutboundDispatch(
                message.channel,
                message.chat_id,
                committed.assistant_content,
                thinking=thinking,
                metadata={"provider_uuid": provider_uuid},
                media=list(committed.media),
            )
        )
        if not sent:
            raise DeliveryError("passive reply delivery was not confirmed")
        return committed.background_tasks


def _inbound_message(payload: Mapping[str, object]) -> InboundMessage:
    timestamp = datetime.fromisoformat(str(payload.get("timestamp") or ""))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    metadata, media = payload.get("metadata"), payload.get("media")
    return InboundMessage(
        str(payload.get("channel") or ""),
        str(payload.get("sender") or ""),
        str(payload.get("chat_id") or ""),
        str(payload.get("content") or ""),
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


__all__ = ["PassiveTurnPipeline"]
