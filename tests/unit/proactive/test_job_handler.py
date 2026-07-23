from datetime import UTC, datetime
from types import SimpleNamespace

from memopilot.delivery.feishu import DeliveryOutcome
from memopilot.proactive.job_handler import ProactiveJobHandlerService
from memopilot.proactive.service import ProactiveOutcome
from memopilot.runtime.engine import TurnInput

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


class _ProactiveService:
    def __init__(self, outcome: ProactiveOutcome) -> None:
        self.outcome = outcome
        self.finalized: list[ProactiveOutcome] = []
        self.failed: list[ProactiveOutcome] = []

    async def execute(self, **kwargs):  # type: ignore[no-untyped-def]
        self.execute_kwargs = kwargs
        return self.outcome

    def finalize_confirmed(self, outcome: ProactiveOutcome, *, confirmed_at: datetime) -> bool:
        del confirmed_at
        self.finalized.append(outcome)
        return True

    def finalize_failed(self, outcome: ProactiveOutcome, *, failed_at: datetime) -> bool:
        del failed_at
        self.failed.append(outcome)
        return True


class _Effects:
    def __init__(self, state: str | None = None, *, run_id: str = "old-run") -> None:
        self.state = state
        self.run_id = run_id
        self.adoptions: list[dict[str, object]] = []

    def get(self, operation_id: str):  # type: ignore[no-untyped-def]
        del operation_id
        return None if self.state is None else SimpleNamespace(state=self.state, run_id=self.run_id)

    def adopt_pending(self, operation_id: str, **kwargs):  # type: ignore[no-untyped-def]
        self.adoptions.append({"operation_id": operation_id, **kwargs})
        self.run_id = str(kwargs["run_id"])
        return True


class _Dispatcher:
    def __init__(self, outcome: DeliveryOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    async def dispatch(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return SimpleNamespace(outcome=self.outcome)


class _Operational:
    def __init__(self) -> None:
        self.enqueued: list[dict[str, object]] = []

    def enqueue_system_job(self, **kwargs):  # type: ignore[no-untyped-def]
        self.enqueued.append(kwargs)
        return SimpleNamespace(job_id="followup", created=True)

    def get_job(self, job_id: str):  # type: ignore[no-untyped-def]
        del job_id
        return SimpleNamespace(activity_version=4)

    def get_activity_version(self, session_key: str) -> int:
        del session_key
        return 4


async def test_confirmed_batched_alert_uses_stable_effect_without_followup() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            trigger_kind="alert",
            message="警报",
            decision_id="decision-1",
            effect_operation_id="effect-1",
            decided_at=NOW,
        )
    )
    dispatcher = _Dispatcher(DeliveryOutcome.CONFIRMED)
    operational = _Operational()
    handler = ProactiveJobHandlerService(
        operational=operational,  # type: ignore[arg-type]
        effects=_Effects(),  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        service_factory=lambda claim, lease: proactive,
    )

    result = await handler.execute(
        payload={"chat_id": "chat-1"},
        claim=SimpleNamespace(job_id="proactive-job", run_id="run-1", session_key="feishu:chat-1"),
        lease=SimpleNamespace(),
        turn=TurnInput("feishu:chat-1", ""),
        now=NOW,
    )

    assert result == "succeeded"
    assert dispatcher.calls[0]["operation_id"] == "effect-1"
    assert proactive.finalized == [proactive.outcome]
    assert operational.enqueued == []


async def test_existing_confirmed_effect_is_finalized_without_cross_run_dispatch() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="恢复消息",
            decision_id="decision-1",
            effect_operation_id="effect-1",
            decided_at=NOW,
        )
    )
    dispatcher = _Dispatcher(DeliveryOutcome.CONFIRMED)
    handler = ProactiveJobHandlerService(
        operational=_Operational(),  # type: ignore[arg-type]
        effects=_Effects("confirmed"),  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        service_factory=lambda claim, lease: proactive,
    )

    result = await handler.execute(
        payload={"chat_id": "chat-1"},
        claim=SimpleNamespace(job_id="new-job", run_id="new-run", session_key="feishu:chat-1"),
        lease=SimpleNamespace(),
        turn=TurnInput("feishu:chat-1", ""),
        now=NOW,
    )

    assert result == "succeeded"
    assert dispatcher.calls == []
    assert proactive.finalized == [proactive.outcome]


async def test_existing_unknown_effect_does_not_blindly_send_from_new_run() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="恢复消息",
            decision_id="decision-1",
            effect_operation_id="effect-1",
            decided_at=NOW,
        )
    )
    dispatcher = _Dispatcher(DeliveryOutcome.CONFIRMED)
    handler = ProactiveJobHandlerService(
        operational=_Operational(),  # type: ignore[arg-type]
        effects=_Effects("unknown"),  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        service_factory=lambda claim, lease: proactive,
    )

    result = await handler.execute(
        payload={"chat_id": "chat-1"},
        claim=SimpleNamespace(job_id="new-job", run_id="new-run", session_key="feishu:chat-1"),
        lease=SimpleNamespace(),
        turn=TurnInput("feishu:chat-1", ""),
        now=NOW,
    )

    assert result == "succeeded"
    assert dispatcher.calls == []
    assert proactive.finalized == []


async def test_same_run_pending_effect_can_resume_dispatch_without_new_identity() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="恢复消息",
            decision_id="decision-1",
            effect_operation_id="effect-1",
            decided_at=NOW,
        )
    )
    dispatcher = _Dispatcher(DeliveryOutcome.CONFIRMED)
    handler = ProactiveJobHandlerService(
        operational=_Operational(),  # type: ignore[arg-type]
        effects=_Effects("pending", run_id="same-run"),  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        service_factory=lambda claim, lease: proactive,
    )

    result = await handler.execute(
        payload={"chat_id": "chat-1"},
        claim=SimpleNamespace(job_id="same-job", run_id="same-run", session_key="feishu:chat-1"),
        lease=SimpleNamespace(),
        turn=TurnInput("feishu:chat-1", ""),
        now=NOW,
    )

    assert result == "succeeded"
    assert len(dispatcher.calls) == 1


async def test_failed_old_run_pending_effect_is_adopted_before_dispatch() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="恢复消息",
            decision_id="decision-1",
            effect_operation_id="effect-1",
            decided_at=NOW,
        )
    )
    effects = _Effects("pending", run_id="failed-run")
    dispatcher = _Dispatcher(DeliveryOutcome.CONFIRMED)
    handler = ProactiveJobHandlerService(
        operational=_Operational(),  # type: ignore[arg-type]
        effects=effects,  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        service_factory=lambda claim, lease: proactive,
    )

    result = await handler.execute(
        payload={"chat_id": "chat-1"},
        claim=SimpleNamespace(job_id="new-job", run_id="new-run", session_key="feishu:chat-1"),
        lease=SimpleNamespace(),
        turn=TurnInput("feishu:chat-1", ""),
        now=NOW,
    )

    assert result == "succeeded"
    assert effects.adoptions[0]["run_id"] == "new-run"
    assert len(dispatcher.calls) == 1


async def test_known_feishu_failure_marks_decision_failed_to_stop_tick_loop() -> None:
    proactive = _ProactiveService(
        ProactiveOutcome(
            "send",
            session_key="feishu:chat-1",
            message="无法发送的提醒",
            decision_id="decision-1",
            effect_operation_id="effect-1",
            decided_at=NOW,
        )
    )
    handler = ProactiveJobHandlerService(
        operational=_Operational(),  # type: ignore[arg-type]
        effects=_Effects(),  # type: ignore[arg-type]
        dispatcher=_Dispatcher(DeliveryOutcome.FAILED),  # type: ignore[arg-type]
        service_factory=lambda claim, lease: proactive,
    )

    result = await handler.execute(
        payload={"chat_id": "chat-1"},
        claim=SimpleNamespace(job_id="proactive-job", run_id="run-1", session_key="feishu:chat-1"),
        lease=SimpleNamespace(),
        turn=TurnInput("feishu:chat-1", ""),
        now=NOW,
    )

    assert result == "failed"
    assert proactive.failed == [proactive.outcome]
