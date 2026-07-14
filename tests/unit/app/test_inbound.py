from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.app.inbound import InboundBridge, OperationalInterruptController
from memopilot.channels.contracts import InboundMessage, MessageBus
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.tasks.operational import InboundCommand, OperationalRepository

NOW = datetime(2026, 7, 14, 10, 0, tzinfo=UTC)


def _repository(tmp_path: Path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


def _message(*, text: str = "你好", event_id: str = "event-1") -> InboundMessage:
    return InboundMessage(
        channel="feishu",
        sender="ou_user",
        chat_id="chat-1",
        content=text,
        timestamp=NOW,
        media=("uploads/a.png",),
        metadata={
            "event_id": event_id,
            "message_id": "message-1",
            "chat_type": "p2p",
            "message_type": "text",
            "open_id": "ou_user",
            "user_id": "",
            "union_id": "",
        },
    )


@pytest.mark.asyncio
async def test_bridge_persists_message_as_p0_job_and_duplicate_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    bridge = InboundBridge(repository)

    first = await bridge.handle(_message())
    duplicate = await bridge.handle(_message(event_id="event-2"))

    assert first.created is True
    assert duplicate.created is False
    assert first.job_id == duplicate.job_id
    job = repository.get_job(first.job_id)
    assert job is not None
    assert job.priority == 0
    payload = json.loads(job.payload_json)
    assert payload["text"] == "你好"
    assert payload["media"] == ["uploads/a.png"]
    assert repository.get_activity_version("feishu:chat-1") == 1


@pytest.mark.asyncio
async def test_bridge_run_once_consumes_message_bus(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bridge = InboundBridge(repository)
    bus = MessageBus()
    await bus.publish_inbound(_message())

    result = await bridge.run_once(bus)

    assert result.created is True
    assert repository.count("agent_jobs") == 1


@pytest.mark.asyncio
async def test_stop_controller_is_idempotent_and_invalidates_previous_activity(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.accept_inbound(
        InboundCommand(
            event_id="event-0",
            message_id="message-0",
            session_key="feishu:chat-1",
            channel="feishu",
            chat_id="chat-1",
            payload={"text": "旧消息"},
            received_at=NOW,
        )
    )
    controller = OperationalInterruptController(repository)
    stop = _message(text="/stop", event_id="stop-event")

    first = await controller.request_interrupt(stop)
    second = await controller.request_interrupt(stop)

    assert first == second
    assert first.message == "已请求中断。"
    assert len(first.provider_uuid) == 36
    assert repository.get_activity_version("feishu:chat-1") == 2
    assert repository.count("session_interrupts") == 1
    assert repository.count("agent_jobs") == 1
