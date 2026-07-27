from datetime import UTC, datetime

import pytest

from memopilot.runtime.background import BackgroundTaskDispatcher
from memopilot.runtime.outbound import OutboundDispatch
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.redis_queue import QueueMessage

NOW = datetime(2026, 7, 27, tzinfo=UTC)
LEASE = SessionLease("feishu:chat-1", "runner-1", 1, "lease-key", "lease-value")


class _Proactive:
    async def execute_task(self, **kwargs) -> str:
        del kwargs
        return "drift"


class _DriftResult:
    outcome = "succeeded"


class _Drift:
    def __init__(self) -> None:
        self.calls = 0

    async def execute_task(self, **kwargs) -> _DriftResult:
        del kwargs
        self.calls += 1
        return _DriftResult()


class _Unused:
    def __getattr__(self, name):
        raise AssertionError(f"不应访问 {name}")


class _ScheduleRepository:
    def __init__(self) -> None:
        self.outcomes: list[str] = []

    def transition_background_schedule(self, execution_id, *, lease, outcome, now):
        del execution_id, lease, now
        self.outcomes.append(outcome)
        return outcome


class _Outbound:
    def __init__(self) -> None:
        self.calls: list[OutboundDispatch] = []

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        self.calls.append(outbound)
        return True


@pytest.mark.asyncio
async def test_proactive_drift_decision_continues_into_drift_runtime() -> None:
    drift = _Drift()
    executor = BackgroundTaskDispatcher(
        _Unused(),  # type: ignore[arg-type]
        repository=_Unused(),  # type: ignore[arg-type]
        outbound=_Unused(),  # type: ignore[arg-type]
        memory_tasks=_Unused(),  # type: ignore[arg-type]
        proactive=_Proactive(),
        drift=drift,
    )
    message = QueueMessage(
        "memopilot:jobs:p2",
        "1-0",
        "proactive.tick:1",
        "proactive.tick",
        2,
        "feishu:chat-1",
        "{}",
    )

    outcome = await executor.execute(
        message,
        payload={"chat_id": "chat-1", "activity_version": 0},
        lease=LEASE,
        now=NOW,
    )

    assert outcome == "succeeded"
    assert drift.calls == 1


@pytest.mark.asyncio
async def test_schedule_dispatches_to_originating_channel() -> None:
    repository = _ScheduleRepository()
    outbound = _Outbound()
    dispatcher = BackgroundTaskDispatcher(
        _Unused(),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        outbound=outbound,
        memory_tasks=_Unused(),  # type: ignore[arg-type]
        proactive=_Unused(),  # type: ignore[arg-type]
        drift=_Unused(),  # type: ignore[arg-type]
    )
    message = QueueMessage(
        "memopilot:jobs:p1",
        "1-0",
        "execution-1",
        "schedule.run",
        1,
        "cli:session-1",
        "{}",
    )

    result = await dispatcher.execute(
        message,
        payload={
            "execution_id": "execution-1",
            "execution_mode": "instant",
            "payload": {"message": "检查 MemoPilot"},
            "channel": "cli",
            "chat_id": "session-1",
        },
        lease=LEASE,
        now=NOW,
    )

    assert result == "succeeded"
    assert outbound.calls == [
        OutboundDispatch(
            channel="cli",
            chat_id="session-1",
            content="检查 MemoPilot",
        )
    ]
    assert repository.outcomes == ["running", "succeeded"]
