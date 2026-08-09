from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest

from memopilot.persistence.conversation import ConversationRepository, StaleActivityError
from memopilot.persistence.migrations import DatabaseKind, migrate_database
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


class _FailingConversation:
    def commit_direct_assistant(self, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("history unavailable")


class _Drift:
    def __init__(self) -> None:
        self.calls = []

    async def execute_task(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)


def _loop(
    proactive: _ProactiveService,
    outbound: _Outbound,
    drift: _Drift | None = None,
    conversation: ConversationRepository | None = None,
) -> ProactiveLoop:
    return ProactiveLoop(
        service_factory=lambda session, activity: proactive,
        outbound=outbound,
        drift=drift or _Drift(),
        conversation=conversation,
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


async def test_confirmed_proactive_send_commits_original_session_metadata(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    conversation = ConversationRepository(database)
    outcome = ProactiveOutcome(
        "send",
        session_key="feishu:chat-1",
        message="这几篇文章值得看",
        decision_id="decision-1",
        evidence_ids=("feed:article-1",),
        decided_at=NOW,
    )

    loop = _loop(_ProactiveService(outcome), _Outbound(True), conversation=conversation)
    await loop.execute_task(
        AgentTask(
            "proactive-1", "proactive.tick", 2, "feishu:chat-1",
            {"channel": "feishu", "chat_id": "chat-1", "activity_version": 4}, NOW,
        ),
        now=NOW,
    )

    record = conversation.list_recent_messages("feishu:chat-1", limit=10)[0]
    assert record.content == "这几篇文章值得看"
    assert record.metadata == {
        "proactive": True,
        "tools_used": ["message_push"],
        "evidence_item_ids": ["feed:article-1"],
        "state_summary_tag": "none",
    }


async def test_failed_proactive_send_does_not_commit_session_message(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    conversation = ConversationRepository(database)
    outcome = ProactiveOutcome(
        "send",
        session_key="feishu:chat-1",
        message="未送达",
        decision_id="decision-1",
        decided_at=NOW,
    )

    with pytest.raises(RuntimeError, match="明确发送成功"):
        loop = _loop(_ProactiveService(outcome), _Outbound(False), conversation=conversation)
        await loop.execute_task(
            AgentTask(
                "proactive-1", "proactive.tick", 2, "feishu:chat-1",
                {"channel": "feishu", "chat_id": "chat-1", "activity_version": 4}, NOW,
            ),
            now=NOW,
        )

    assert conversation.list_recent_messages("feishu:chat-1", limit=10) == ()


async def test_proactive_replay_keeps_one_direct_assistant_message(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    conversation = ConversationRepository(database)
    outcome = ProactiveOutcome(
        "send",
        session_key="feishu:chat-1",
        message="可重放",
        decision_id="decision-1",
        decided_at=NOW,
    )
    loop = _loop(_ProactiveService(outcome), _Outbound(True), conversation=conversation)
    task = AgentTask(
        "proactive-1", "proactive.tick", 2, "feishu:chat-1",
        {"channel": "feishu", "chat_id": "chat-1", "activity_version": 4}, NOW,
    )

    await loop.execute_task(task, now=NOW)
    await loop.execute_task(task, now=NOW)

    assert len(conversation.list_recent_messages("feishu:chat-1", limit=10)) == 1


def test_direct_assistant_commit_is_idempotent_and_rejects_changed_replay(tmp_path) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    conversation = ConversationRepository(database)
    arguments = {
        "session_key": "feishu:chat-1",
        "channel": "feishu",
        "chat_id": "chat-1",
        "content": "可重放",
        "media": (),
        "timestamp": NOW,
        "source_ref": "proactive:decision-1",
        "metadata": {"proactive": True},
    }

    assert conversation.commit_direct_assistant(**arguments).inserted is True
    assert conversation.commit_direct_assistant(**arguments).inserted is False
    with pytest.raises(ValueError, match="inconsistent"):
        conversation.commit_direct_assistant(**(arguments | {"content": "不同内容"}))


async def test_history_commit_failure_does_not_finalize_proactive_send() -> None:
    service = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="待保存",
            decision_id="decision-1",
            decided_at=NOW,
        )
    )
    loop = _loop(service, _Outbound(True), conversation=_FailingConversation())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="history unavailable"):
        await loop.execute_task(
            AgentTask(
                "proactive-1", "proactive.tick", 2, "feishu:chat-1",
                {"channel": "feishu", "chat_id": "chat-1", "activity_version": 4}, NOW,
            ),
            now=NOW,
        )

    assert service.finalized == []


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
