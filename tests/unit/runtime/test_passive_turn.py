from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.extensions.events import EventBus
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.engine import TurnResult
from memopilot.runtime.outbound import DeliveryError, OutboundDispatch
from memopilot.runtime.passive_turn import PassiveTurnPipeline
from memopilot.runtime.react import ReActResult
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository

NOW = datetime(2026, 7, 30, tzinfo=UTC)
LEASE = SessionLease("cli:chat", "agent", 1, "key", "value")


class _Runtime:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, turn: object, **kwargs: object) -> TurnResult:
        del turn, kwargs
        self.calls += 1
        react = ReActResult("reply", (), 1, (), "completed")
        return TurnResult("reply", (ChatMessage.assistant(content="reply"),), react, (), ())


class _Outbound:
    def __init__(self) -> None:
        self.sent: list[OutboundDispatch] = []
        self.ok = False

    async def dispatch(self, dispatch: OutboundDispatch) -> bool:
        self.sent.append(dispatch)
        return self.ok


@pytest.mark.asyncio
async def test_replay_reuses_committed_reply_and_stable_send_id(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    message = InboundMessage(
        "cli", "user", "chat", "hi", timestamp=NOW, metadata={"message_id": "m1"}
    )
    repository.record_inbound_activity(message)
    repository.allocate_fence(message.session_key, owner_id="agent", now=NOW)
    outbound = _Outbound()
    pipeline = PassiveTurnPipeline(
        _Runtime(),
        repository=repository,
        outbound=outbound,
        event_bus=EventBus(),
        history_limit=12,
    )
    task = AgentTask(
        "t1",
        "passive.turn",
        0,
        message.session_key,
        {
            "channel": "cli",
            "sender": "user",
            "chat_id": "chat",
            "content": "hi",
            "timestamp": NOW.isoformat(),
            "metadata": {"message_id": "m1"},
        },
        NOW,
    )

    with pytest.raises(DeliveryError):
        await pipeline.execute_task(task, lease=LEASE, now=NOW)
    outbound.ok = True
    await pipeline.execute_task(task, lease=LEASE, now=NOW)

    assert pipeline._runtime.calls == 1
    assert [item.metadata["provider_uuid"] for item in outbound.sent] == [
        outbound.sent[0].metadata["provider_uuid"],
        outbound.sent[0].metadata["provider_uuid"],
    ]
