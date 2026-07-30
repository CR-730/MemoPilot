from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from memopilot.bus.events import InboundMessage, TurnCommitted
from memopilot.extensions.events import EventBus
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.background import CoreRunner
from memopilot.runtime.common_tools.message_push import MessagePushTool
from memopilot.runtime.contracts import ChatMessage, FunctionCall
from memopilot.runtime.engine import TurnResult
from memopilot.runtime.outbound import (
    DeliveryError,
    OutboundDispatch,
    PushToolOutboundPort,
)
from memopilot.runtime.react import ReActResult, ToolCallRecord
from memopilot.runtime.tools import ToolObservation
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.redis_queue import QueueMessage

NOW = datetime(2026, 7, 28, tzinfo=UTC)
LEASE = SessionLease("feishu:chat-1", "agent-1", 1, "lease-key", "lease-value")


class _Runtime:
    def __init__(self, media: tuple[str, ...] = ()) -> None:
        self.calls = 0
        self.media = media

    async def run(self, turn: object, **kwargs: object) -> TurnResult:
        del turn, kwargs
        self.calls += 1
        react = ReActResult("旧回复", (), 1, (), "completed")
        return TurnResult("旧回复", (), react, (), (), self.media)


class _Outbound:
    def __init__(self, events: list[str]) -> None:
        self.results = iter((False, True))
        self.calls: list[OutboundDispatch] = []
        self.events = events

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        self.events.append("send")
        self.calls.append(outbound)
        return next(self.results)


class _Unused:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"不应访问 {name}")


def _runner(
    repository: OperationalRepository,
    events: list[str],
) -> tuple[CoreRunner, _Runtime, _Outbound, EventBus]:
    runtime = _Runtime()
    outbound = _Outbound(events)
    event_bus = EventBus()
    return (
        CoreRunner(
            runtime,  # type: ignore[arg-type]
            repository=repository,
            outbound=outbound,
            memory_tasks=_Unused(),  # type: ignore[arg-type]
            proactive=_Unused(),  # type: ignore[arg-type]
            drift=_Unused(),  # type: ignore[arg-type]
            event_bus=event_bus,
        ),
        runtime,
        outbound,
        event_bus,
    )


@pytest.mark.asyncio
async def test_committed_turn_retry_reuses_reply_model_and_event_once(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    message_time = NOW
    inbound = {
        "channel": "feishu",
        "sender": "user",
        "chat_id": "chat-1",
        "content": "你好",
        "timestamp": message_time.isoformat(),
        "media": [],
        "metadata": {"message_id": "message-1"},
    }
    repository.record_inbound_activity(
        InboundMessage(
            "feishu",
            "user",
            "chat-1",
            "你好",
            timestamp=message_time,
            metadata={"message_id": "message-1"},
        )
    )
    repository.allocate_fence("feishu:chat-1", owner_id="agent-1", now=NOW)
    order: list[str] = []
    runner, runtime, outbound, event_bus = _runner(repository, order)
    events: list[object] = []
    event_bus.on(
        TurnCommitted,
        lambda event: (events.append(event), order.append("event"))[0],
        observer=True,
    )
    payload = inbound
    message = QueueMessage(
        "memopilot:tasks:p0",
        "1-0",
        "task-1",
        "passive.turn",
        0,
        "feishu:chat-1",
        json.dumps(payload),
    )

    with pytest.raises(RuntimeError, match="明确发送成功"):
        await runner.execute(message, payload=payload, lease=LEASE, now=NOW)
    await runner.execute(message, payload=payload, lease=LEASE, now=NOW)

    assert runtime.calls == 1
    assert [call.content for call in outbound.calls] == ["旧回复", "旧回复"]
    assert len(events) == 1
    assert order == ["event", "send", "send"]
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_runner_persists_react_tool_chain_for_next_history(tmp_path: Path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    inbound = InboundMessage(
        "feishu", "user", "chat-1", "列出目录", timestamp=NOW,
        metadata={"message_id": "message-tools"},
    )
    repository.record_inbound_activity(inbound)
    repository.allocate_fence(inbound.session_key, owner_id="agent-1", now=NOW)
    call = FunctionCall("call-1", "list_dir", {"path": "."})

    class Runtime(_Runtime):
        async def run(self, turn: object, **kwargs: object) -> TurnResult:
            del turn, kwargs
            self.calls += 1
            react = ReActResult(
                "已查看",
                (
                    ChatMessage.assistant(content="我来查看", tool_calls=(call,)),
                    ChatMessage.tool(call_id="call-1", name="list_dir", content="目录"),
                    ChatMessage.assistant(content="已查看"),
                ),
                1,
                (ToolCallRecord(1, call, ToolObservation("call-1", "list_dir", True, "目录")),),
                "completed",
            )
            return TurnResult("已查看", (), react, (), ())

    runtime = Runtime()
    outbound = _Outbound([])
    event_bus = EventBus()
    runner = CoreRunner(
        runtime,  # type: ignore[arg-type]
        repository=repository,
        outbound=outbound,
        memory_tasks=_Unused(),  # type: ignore[arg-type]
        proactive=_Unused(),  # type: ignore[arg-type]
        drift=_Unused(),  # type: ignore[arg-type]
        event_bus=event_bus,
    )
    payload = {
        "channel": "feishu", "sender": "user", "chat_id": "chat-1", "content": "列出目录",
        "timestamp": NOW.isoformat(), "media": [], "metadata": {"message_id": "message-tools"},
    }
    message = QueueMessage("p0", "1-0", "task-tools", "passive.turn", 0, inbound.session_key, json.dumps(payload))

    with pytest.raises(RuntimeError):
        await runner.execute(message, payload=payload, lease=LEASE, now=NOW)

    saved = repository.list_recent_messages(inbound.session_key, limit=2)[1]
    assert saved.tool_chain[0]["calls"] == [{"call_id": "call-1", "name": "list_dir", "arguments": {"path": "."}, "result": "目录"}]
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_committed_turn_retry_restores_media_with_same_provider_uuid(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    inbound_message = InboundMessage(
        "feishu",
        "user",
        "chat-1",
        "来个表情",
        timestamp=NOW,
        metadata={"message_id": "message-media"},
    )
    repository.record_inbound_activity(inbound_message)
    repository.allocate_fence("feishu:chat-1", owner_id="agent-1", now=NOW)
    payload = {
        "channel": "feishu",
        "sender": "user",
        "chat_id": "chat-1",
        "content": "来个表情",
        "timestamp": NOW.isoformat(),
        "media": [],
        "metadata": {"message_id": "message-media"},
    }
    message = QueueMessage(
        "memopilot:tasks:p0",
        "1-0",
        "task-media",
        "passive.turn",
        0,
        "feishu:chat-1",
        json.dumps(payload),
    )
    text_uuids: list[str | None] = []
    image_uuids: list[str] = []

    async def send_text(
        chat_id: str,
        content: str,
        *,
        provider_uuid: str | None = None,
    ) -> None:
        del chat_id, content
        text_uuids.append(provider_uuid)

    async def send_image(
        chat_id: str,
        image: str,
        *,
        provider_uuid: str,
    ) -> None:
        del chat_id, image
        image_uuids.append(provider_uuid)
        if len(image_uuids) == 1:
            raise RuntimeError("image failed")

    push = MessagePushTool()
    push.register_channel("feishu", text=send_text, image=send_image)
    runtime = _Runtime(("meme.png",))
    event_bus = EventBus()
    runner = CoreRunner(
        runtime,  # type: ignore[arg-type]
        repository=repository,
        outbound=PushToolOutboundPort(push),
        memory_tasks=_Unused(),  # type: ignore[arg-type]
        proactive=_Unused(),  # type: ignore[arg-type]
        drift=_Unused(),  # type: ignore[arg-type]
        event_bus=event_bus,
    )

    with pytest.raises(DeliveryError, match="明确发送成功"):
        await runner.execute(message, payload=payload, lease=LEASE, now=NOW)
    await runner.execute(message, payload=payload, lease=LEASE, now=NOW)

    assert runtime.calls == 1
    assert len(text_uuids) == 2
    assert text_uuids[0] == text_uuids[1]
    assert len(image_uuids) == 2
    assert image_uuids[0] == image_uuids[1]
    await event_bus.aclose()
