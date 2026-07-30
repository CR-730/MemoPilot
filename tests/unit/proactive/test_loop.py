from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest

from memopilot.persistence.conversation import StaleActivityError
from memopilot.proactive.loop import ProactiveLoop
from memopilot.proactive.service import ProactiveOutcome
from memopilot.runtime.outbound import OutboundDispatch
from memopilot.tasks.agent_task import AgentTask

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


class _ProactiveService:
    def __init__(self, outcome: ProactiveOutcome, *, stale_before_send: bool = False) -> None:
        self.outcome = outcome
        self.stale_before_send = stale_before_send
        self.finalized: list[ProactiveOutcome] = []
        self.failed: list[ProactiveOutcome] = []

    async def execute(self, **kwargs):  # type: ignore[no-untyped-def]
        self.execute_kwargs = kwargs
        return self.outcome

    def finalize_confirmed(self, outcome, *, confirmed_at=None):  # type: ignore[no-untyped-def]
        del confirmed_at
        self.finalized.append(outcome)
        return True

    def assert_current(self) -> None:
        if self.stale_before_send:
            raise StaleActivityError("activity changed")

    def finalize_failed(self, outcome, *, failed_at=None):  # type: ignore[no-untyped-def]
        del failed_at
        self.failed.append(outcome)
        return True


class _Outbound:
    def __init__(self, sent: bool) -> None:
        self.sent = sent
        self.calls: list[OutboundDispatch] = []

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        self.calls.append(outbound)
        return self.sent


class _Drift:
    def __init__(self) -> None:
        self.calls = []

    async def execute_task(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)


def _loop(
    proactive: _ProactiveService, outbound: _Outbound, drift: _Drift | None = None
) -> ProactiveLoop:
    return ProactiveLoop(
        service_factory=lambda session, activity: proactive,
        outbound=outbound,
        drift=drift or _Drift(),
    )


async def test_proactive_reply_dispatches_then_commits_decision() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            trigger_kind="alert",
            message="警报",
            decision_id="decision-1",
            decided_at=NOW,
        )
    )
    outbound = _Outbound(True)

    result = await _loop(proactive, outbound).execute_task(
        AgentTask(
            "proactive-1", "proactive.tick", 2, "feishu:chat-1",
            {"channel": "cli", "chat_id": "chat-1", "activity_version": 4}, NOW,
        ),
        now=NOW,
    )

    assert result == ()
    assert len(outbound.calls) == 1
    assert outbound.calls[0].channel == "cli"
    assert outbound.calls[0].chat_id == "chat-1"
    assert outbound.calls[0].content == "警报"
    assert outbound.calls[0].metadata["provider_uuid"] == str(
        uuid5(NAMESPACE_URL, "memopilot:proactive:decision-1")
    )
    assert proactive.finalized == [proactive.outcome]


async def test_proactive_send_failure_stays_retryable() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="无法发送的提醒",
            decision_id="decision-1",
            decided_at=NOW,
        )
    )

    with pytest.raises(RuntimeError, match="明确发送成功"):
        await _loop(proactive, _Outbound(False)).execute_task(
            AgentTask(
                "proactive-1", "proactive.tick", 2, "feishu:chat-1",
                {"channel": "cli", "chat_id": "chat-1", "activity_version": 4}, NOW,
            ),
            now=NOW,
        )

    assert proactive.failed == []


async def test_activity_change_before_send_blocks_dispatch() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="过期提醒",
            decision_id="decision-1",
            decided_at=NOW,
        ),
        stale_before_send=True,
    )
    outbound = _Outbound(True)

    with pytest.raises(StaleActivityError, match="activity changed"):
        await _loop(proactive, outbound).execute_task(
            AgentTask(
                "proactive-1",
                "proactive.tick",
                2,
                "feishu:chat-1",
                {"channel": "cli", "chat_id": "chat-1", "activity_version": 4},
                NOW,
            ),
            now=NOW,
        )

    assert outbound.calls == []


@pytest.mark.asyncio
async def test_tick_drift_and_direct_drift_use_same_runner() -> None:
    drift = _Drift()
    service = _ProactiveService(ProactiveOutcome("drift", session_key="feishu:chat-1"))
    loop = _loop(service, _Outbound(True), drift)
    payload = {"channel": "cli", "chat_id": "chat-1", "activity_version": 4}
    for kind in ("proactive.tick", "drift.run"):
        await loop.execute_task(
            AgentTask(kind, kind, 2, "feishu:chat-1", payload, NOW),
            now=NOW,
        )
    assert [call["task_id"] for call in drift.calls] == ["proactive.tick", "drift.run"]
