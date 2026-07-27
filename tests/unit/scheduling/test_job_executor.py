from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.delivery.feishu import DeliveryOutcome
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.proactive.store import ProactiveRepository
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.engine import TurnInput, TurnResult
from memopilot.runtime.react import ReActResult
from memopilot.scheduling.job_executor import SystemJobRouter

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _result(reply: str = "运行结果", *, infrastructure_error: bool = False) -> TurnResult:
    react = ReActResult(
        reply=reply,
        messages=(),
        iterations=1,
        tool_chain=(),
        exit_reason="infrastructure_error" if infrastructure_error else "completed",
        infrastructure_error=infrastructure_error,
    )
    return TurnResult(reply, (), react, (), ())


class _Repository:
    def __init__(self) -> None:
        self.fence_checks = 0
        self.transitions: list[tuple[str, str]] = []

    def assert_current_fence(self, lease: Any) -> None:
        self.fence_checks += 1

    def assert_current_fence_and_activity(
        self,
        lease: Any,
        *,
        expected_activity_version: int,
    ) -> None:
        del lease, expected_activity_version
        self.fence_checks += 1

    def get_job(self, job_id: str):
        del job_id
        return SimpleNamespace(activity_version=4)

    def transition_scheduled_execution(
        self,
        execution_id: str,
        *,
        job_id: str,
        lease: Any,
        outcome: str,
        now: datetime,
    ) -> str:
        self.transitions.append((execution_id, outcome))
        return outcome


class _Runtime:
    def __init__(self, result: TurnResult | None = None) -> None:
        self.calls: list[tuple[TurnInput, dict[str, Any]]] = []
        self.result = result or _result()

    async def run(self, turn: TurnInput, **kwargs: Any) -> TurnResult:
        self.calls.append((turn, kwargs))
        return self.result


class _FailingRuntime(_Runtime):
    async def run(self, turn: TurnInput, **kwargs: Any) -> TurnResult:
        raise RuntimeError("runtime failed")


class _FinishingDriftRuntime(_Runtime):
    async def run(self, turn: TurnInput, **kwargs: Any) -> TurnResult:
        self.calls.append((turn, kwargs))
        finished = await kwargs["tools"].execute(
            FunctionCall(
                "finish",
                "finish_drift",
                {
                    "skill_used": "research",
                    "one_line": "完成后台检查",
                    "next": "等待下次空闲",
                    "message_result": "silent",
                },
            )
        )
        assert finished.ok
        return self.result


class _BrokenDriftAudit:
    def mark_drift_started(self, **kwargs: Any) -> None:
        del kwargs

    def complete_drift(self, **kwargs: Any) -> None:
        del kwargs
        raise KeyError("audit failed")


class _Dispatcher:
    def __init__(self, outcome: DeliveryOutcome = DeliveryOutcome.CONFIRMED) -> None:
        self.outcome = outcome
        self.calls: list[str] = []

    async def dispatch(self, *, claim: Any, lease: Any, text: str):
        self.calls.append(text)
        return SimpleNamespace(outcome=self.outcome)


class _FailingDispatcher(_Dispatcher):
    async def dispatch(self, *, claim: Any, lease: Any, text: str):
        raise RuntimeError("dispatch failed")


class _ProactiveHandler:
    def __init__(self, outcome: str = "succeeded") -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.outcome


class _DriftSelector:
    def __init__(self, selected: str = "research") -> None:
        self.selected = selected
        self.calls = 0

    async def select(self) -> str:
        self.calls += 1
        return self.selected


def _identity():
    claim = SimpleNamespace(
        job_id="job-system",
        run_id="run-system",
        session_key="feishu:chat-1",
        owner_id="worker-1",
        fencing_epoch=1,
    )
    lease = SimpleNamespace(
        session_key="feishu:chat-1",
        owner_id="worker-1",
        epoch=1,
    )
    return claim, lease


@pytest.mark.asyncio
async def test_schedule_instant_sends_fixed_text_without_runtime_call() -> None:
    repository = _Repository()
    runtime = _Runtime()
    dispatcher = _Dispatcher()
    claim, lease = _identity()
    router = SystemJobRouter(repository, runtime, dispatcher=dispatcher)  # type: ignore[arg-type]

    result = await router.execute(
        kind="schedule.run",
        payload={
            "execution_id": "exec-1",
            "execution_mode": "instant",
            "payload": {"message": "喝水"},
        },
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result.outcome == "succeeded"
    assert runtime.calls == []
    assert dispatcher.calls == ["喝水"]
    assert repository.transitions == [("exec-1", "running"), ("exec-1", "succeeded")]


@pytest.mark.asyncio
async def test_schedule_agent_runs_prompt_then_dispatches_reply() -> None:
    repository = _Repository()
    runtime = _Runtime(_result("今天晴"))
    dispatcher = _Dispatcher()
    claim, lease = _identity()
    router = SystemJobRouter(repository, runtime, dispatcher=dispatcher)  # type: ignore[arg-type]

    result = await router.execute(
        kind="schedule.run",
        payload={
            "execution_id": "exec-2",
            "execution_mode": "agent",
            "payload": {"prompt": "查询天气"},
        },
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result.outcome == "succeeded"
    assert runtime.calls[0][0].content == "查询天气"
    assert runtime.calls[0][0].prompt_scope == "scheduled"
    assert dispatcher.calls == ["今天晴"]


@pytest.mark.asyncio
async def test_drift_selects_skill_then_runs_without_sending_final_text(
    tmp_path: Path,
) -> None:
    repository = _Repository()
    runtime = _FinishingDriftRuntime(_result("后台结论"))
    dispatcher = _Dispatcher()
    database = tmp_path / "proactive.db"
    migrate_database(database, DatabaseKind.PROACTIVE)
    drift_audit = ProactiveRepository(database)
    claim, lease = _identity()
    selector = _DriftSelector()
    router = SystemJobRouter(
        repository,
        runtime,
        dispatcher=dispatcher,
        drift_selector=selector,
        drift_repository=drift_audit,
    )  # type: ignore[arg-type]

    result = await router.execute(
        kind="drift.run",
        payload={},
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result.outcome == "succeeded"
    assert selector.calls == 1
    assert runtime.calls[0][0].content == "$research"
    assert runtime.calls[0][0].prompt_scope == "background"
    assert dispatcher.calls == []
    audit = drift_audit.list_drift_history("feishu:chat-1")[0]
    assert audit["outcome"] == "succeeded"
    assert audit["reason"] == "silent"


@pytest.mark.asyncio
async def test_drift_runtime_error_is_not_masked_by_audit_failure() -> None:
    repository = _Repository()
    claim, lease = _identity()
    router = SystemJobRouter(
        repository,
        _FailingRuntime(),
        drift_selector=_DriftSelector(),
        drift_repository=_BrokenDriftAudit(),  # type: ignore[arg-type]
    )  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="runtime failed"):
        await router.execute(
            kind="drift.run",
            payload={},
            claim=claim,
            lease=lease,
            turn=TurnInput("feishu:chat-1", "", received_at=NOW),
            now=NOW,
        )


@pytest.mark.asyncio
async def test_drift_without_finish_is_failed() -> None:
    repository = _Repository()
    claim, lease = _identity()
    router = SystemJobRouter(
        repository,
        _Runtime(_result("只输出文字，没有 finish_drift")),
        drift_selector=_DriftSelector(),
    )  # type: ignore[arg-type]

    result = await router.execute(
        kind="drift.run",
        payload={},
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result.outcome == "failed"


@pytest.mark.asyncio
async def test_proactive_is_delegated_to_injected_handler() -> None:
    repository = _Repository()
    proactive = _ProactiveHandler("cancelled")
    claim, lease = _identity()
    router = SystemJobRouter(
        repository,
        _Runtime(),
        dispatcher=_Dispatcher(),  # type: ignore[arg-type]
        proactive_handler=proactive,  # type: ignore[arg-type]
    )

    result = await router.execute(
        kind="proactive.tick",
        payload={"bucket": 1},
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result.outcome == "cancelled"
    assert proactive.calls[0]["payload"] == {"bucket": 1}


@pytest.mark.asyncio
async def test_proactive_drift_continues_in_same_job_and_lease() -> None:
    repository = _Repository()
    runtime = _FinishingDriftRuntime(_result("后台结论"))
    proactive = _ProactiveHandler("drift")
    selector = _DriftSelector()
    claim, lease = _identity()
    router = SystemJobRouter(
        repository,
        runtime,
        dispatcher=_Dispatcher(),  # type: ignore[arg-type]
        proactive_handler=proactive,  # type: ignore[arg-type]
        drift_selector=selector,
    )

    result = await router.execute(
        kind="proactive.tick",
        payload={"bucket": 1},
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result.outcome == "succeeded"
    assert selector.calls == 1
    assert runtime.calls[0][0].prompt_scope == "background"


@pytest.mark.asyncio
async def test_schedule_delivery_needs_review_updates_execution_state() -> None:
    repository = _Repository()
    claim, lease = _identity()
    router = SystemJobRouter(
        repository,
        _Runtime(),
        dispatcher=_Dispatcher(DeliveryOutcome.NEEDS_REVIEW),  # type: ignore[arg-type]
    )

    result = await router.execute(
        kind="schedule.run",
        payload={
            "execution_id": "exec-review",
            "execution_mode": "instant",
            "payload": {"message": "提醒"},
        },
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result.outcome == "needs_review"
    assert repository.transitions[-1] == ("exec-review", "needs_review")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "dispatcher", "mode", "task_payload"),
    [
        (_FailingRuntime(), _Dispatcher(), "agent", {"prompt": "查询天气"}),
        (_Runtime(), _FailingDispatcher(), "instant", {"message": "提醒"}),
    ],
)
async def test_schedule_exception_marks_running_execution_failed(
    runtime: _Runtime,
    dispatcher: _Dispatcher,
    mode: str,
    task_payload: dict[str, str],
) -> None:
    repository = _Repository()
    claim, lease = _identity()
    router = SystemJobRouter(repository, runtime, dispatcher=dispatcher)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        await router.execute(
            kind="schedule.run",
            payload={
                "execution_id": "exec-failed",
                "execution_mode": mode,
                "payload": task_payload,
            },
            claim=claim,
            lease=lease,
            turn=TurnInput("feishu:chat-1", "", received_at=NOW),
            now=NOW,
        )

    assert repository.transitions == [
        ("exec-failed", "running"),
        ("exec-failed", "failed"),
    ]
