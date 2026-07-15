from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memopilot.delivery.effects import EffectRecord
from memopilot.delivery.feishu import DeliveryOutcome, DeliveryResult
from memopilot.persistence.migrations import DatabaseKind, connect_database, migrate_database
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import ToolRegistry
from memopilot.runtime.worker import RuntimeJobExecutor, TurnInterrupted
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import (
    InboundCommand,
    InterruptCommand,
    LostLeaseError,
    OperationalRepository,
    RunClaim,
    TurnInterruptSnapshot,
)

NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


class _Provider(ChatProvider):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        return ModelResponse(content="阶段二完成", tool_calls=(), finish_reason="stop")


def _claimed_run(
    tmp_path: Path,
) -> tuple[OperationalRepository, SessionLease, RunClaim]:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    accepted = repository.accept_inbound(
        InboundCommand(
            event_id="event-1",
            message_id="message-1",
            session_key="fake:1",
            channel="fake",
            chat_id="1",
            payload={"text": "hello"},
            received_at=NOW,
        )
    )
    epoch = repository.allocate_fence("fake:1", owner_id="worker-a", now=NOW)
    lease = SessionLease(
        session_key="fake:1",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="test:lease",
        redis_value=f"worker-a|epoch|{epoch}",
    )
    claim = repository.claim_job(accepted.job_id, lease=lease, now=NOW)
    assert claim is not None
    return repository, lease, claim


async def test_claimed_job_completes_runtime_and_persists_ordered_steps(
    tmp_path: Path,
) -> None:
    repository, lease, claim = _claimed_run(tmp_path)
    runtime = AgentRuntime(_Provider(), tools=ToolRegistry())

    result = await RuntimeJobExecutor(repository, runtime, clock=lambda: NOW).execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key="fake:1", content="hello"),
        now=NOW,
    )

    assert result.reply == "阶段二完成"
    job = repository.get_job(claim.job_id)
    assert job is not None
    assert job.state == "succeeded"
    steps = repository.list_steps(claim.run_id)
    assert [step.step_index for step in steps] == list(range(len(steps)))
    assert {step.phase for step in steps} >= {
        "before_turn",
        "before_reasoning",
        "prompt_render",
        "before_step",
        "after_step",
        "after_reasoning",
        "after_turn",
    }


async def test_executor_records_actual_completion_time_not_claim_time(tmp_path: Path) -> None:
    repository, lease, claim = _claimed_run(tmp_path)
    current = [NOW]

    class _AdvancingProvider(ChatProvider):
        async def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSchema],
        ) -> ModelResponse:
            current[0] = NOW + timedelta(seconds=8)
            return ModelResponse(content="完成", tool_calls=())

    await RuntimeJobExecutor(
        repository,
        AgentRuntime(_AdvancingProvider(), ToolRegistry()),
        clock=lambda: current[0],
    ).execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key="fake:1", content="hello"),
        now=NOW,
    )

    with connect_database(repository.database) as connection:
        finished_at = connection.execute(
            "SELECT finished_at FROM runs WHERE run_id = ?", (claim.run_id,)
        ).fetchone()["finished_at"]

    assert finished_at == (NOW + timedelta(seconds=8)).isoformat()


def test_old_fencing_epoch_cannot_append_runtime_step(tmp_path: Path) -> None:
    repository, stale_lease, claim = _claimed_run(tmp_path)
    repository.allocate_fence("fake:1", owner_id="worker-b", now=NOW)

    with pytest.raises(LostLeaseError):
        repository.append_step(
            claim.run_id,
            lease=stale_lease,
            phase="before_step",
            step_type="model",
            state="succeeded",
            now=NOW,
        )


def test_other_session_lease_cannot_write_or_finish_run(tmp_path: Path) -> None:
    repository, _, claim = _claimed_run(tmp_path)
    accepted = repository.accept_inbound(
        InboundCommand(
            event_id="event-2",
            message_id="message-2",
            session_key="fake:2",
            channel="fake",
            chat_id="2",
            payload={"text": "other"},
            received_at=NOW,
        )
    )
    assert accepted.job_id != claim.job_id
    epoch = repository.allocate_fence("fake:2", owner_id="worker-a", now=NOW)
    other_lease = SessionLease(
        session_key="fake:2",
        owner_id="worker-a",
        epoch=epoch,
        redis_key="test:other",
        redis_value=f"worker-a|epoch|{epoch}",
    )
    assert other_lease.epoch == claim.fencing_epoch

    with pytest.raises(LostLeaseError):
        repository.append_step(
            claim.run_id,
            lease=other_lease,
            phase="before_step",
            step_type="model",
            state="succeeded",
            now=NOW,
        )
    with pytest.raises(LostLeaseError):
        repository.finish_job(
            claim.run_id,
            lease=other_lease,
            outcome="succeeded",
            now=NOW,
        )


async def test_provider_fallback_is_persisted_as_failed_job(tmp_path: Path) -> None:
    repository, lease, claim = _claimed_run(tmp_path)
    runtime = AgentRuntime(_FailingProvider(), tools=ToolRegistry())

    result = await RuntimeJobExecutor(repository, runtime, clock=lambda: NOW).execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key="fake:1", content="hello"),
        now=NOW,
    )

    assert result.react.infrastructure_error == "TimeoutError"
    job = repository.get_job(claim.job_id)
    assert job is not None
    assert job.state == "failed"
    assert any(step.state == "failed" for step in repository.list_steps(claim.run_id))


async def test_executor_rejects_turn_from_another_session(tmp_path: Path) -> None:
    repository, lease, claim = _claimed_run(tmp_path)
    runtime = AgentRuntime(_Provider(), tools=ToolRegistry())

    with pytest.raises(ValueError, match="TurnInput"):
        await RuntimeJobExecutor(repository, runtime, clock=lambda: NOW).execute(
            claim=claim,
            lease=lease,
            turn=TurnInput(session_key="fake:2", content="other session"),
            now=NOW,
        )

    assert repository.list_steps(claim.run_id) == ()


async def test_targeted_interrupt_cancels_runtime_and_persists_resume_snapshot(
    tmp_path: Path,
) -> None:
    repository, lease, claim = _claimed_run(tmp_path)

    class _BlockingProvider(ChatProvider):
        async def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSchema],
        ) -> ModelResponse:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(_BlockingProvider(), tools=ToolRegistry()),
        clock=lambda: NOW,
    )
    task = asyncio.create_task(
        executor.execute(
            claim=claim,
            lease=lease,
            turn=TurnInput(session_key="fake:1", content="查天气后发邮件"),
            now=NOW,
        )
    )
    await asyncio.sleep(0)
    repository.request_interrupt(
        InterruptCommand(
            event_id="stop-event",
            message_id="stop-message",
            session_key="fake:1",
            channel="fake",
            chat_id="1",
            requested_at=NOW,
        )
    )
    task.cancel()

    with pytest.raises(TurnInterrupted):
        await task

    assert repository.get_job(claim.job_id).state == "cancelled"  # type: ignore[union-attr]
    next_job = repository.accept_inbound(
        InboundCommand(
            event_id="event-2",
            message_id="message-2",
            session_key="fake:1",
            channel="fake",
            chat_id="1",
            payload={"text": "继续，但改发给小王"},
            received_at=NOW + timedelta(seconds=1),
        )
    )
    resumed = repository.reserve_interrupt_snapshot(
        "fake:1", job_id=next_job.job_id, now=NOW + timedelta(seconds=1)
    )
    assert resumed is not None
    assert resumed.original_message == "查天气后发邮件"


async def test_failed_resumed_delivery_releases_snapshot_instead_of_consuming_it(
    tmp_path: Path,
) -> None:
    repository, lease, source_claim = _claimed_run(tmp_path)
    repository.request_interrupt(
        InterruptCommand(
            event_id="stop-event",
            message_id="stop-message",
            session_key="fake:1",
            channel="fake",
            chat_id="1",
            requested_at=NOW,
        )
    )
    snapshot_id = repository.finish_interrupted_run(
        source_claim.run_id,
        lease=lease,
        snapshot=TurnInterruptSnapshot(original_message="原任务"),
        now=NOW,
    )
    resumed_job = repository.accept_inbound(
        InboundCommand(
            event_id="resume-event",
            message_id="resume-message",
            session_key="fake:1",
            channel="fake",
            chat_id="1",
            payload={"text": "继续", "chat_id": "1"},
            received_at=NOW + timedelta(seconds=1),
        )
    )
    reserved = repository.reserve_interrupt_snapshot(
        "fake:1", job_id=resumed_job.job_id, now=NOW + timedelta(seconds=1)
    )
    assert reserved is not None
    resumed_claim = repository.claim_job(
        resumed_job.job_id, lease=lease, now=NOW + timedelta(seconds=1)
    )
    assert resumed_claim is not None

    class _FailedDispatcher:
        def create_live_progress(self, **_: object) -> None:
            return None

        async def dispatch(self, **_: object):  # type: ignore[no-untyped-def]
            return DeliveryResult(
                DeliveryOutcome.FAILED,
                EffectRecord(
                    operation_id="failed",
                    run_id=resumed_claim.run_id,
                    session_key="fake:1",
                    channel="fake",
                    chat_id="1",
                    payload_json='{"text":""}',
                    payload_hash="",
                    provider_uuid="",
                    expected_activity_version=0,
                    state="cancelled",
                    owner_id=lease.owner_id,
                    fencing_epoch=lease.epoch,
                    message_id=None,
                    first_requested_at=None,
                    cancel_on_activity=False,
                ),
            )

    await RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), tools=ToolRegistry()),
        final_response_dispatcher=_FailedDispatcher(),  # type: ignore[arg-type]
        clock=lambda: NOW + timedelta(seconds=2),
    ).execute(
        claim=resumed_claim,
        lease=lease,
        turn=TurnInput(
            session_key="fake:1",
            content="继续",
            resume_snapshot_id=snapshot_id,
        ),
        now=NOW + timedelta(seconds=1),
    )

    with connect_database(repository.database) as connection:
        row = connection.execute(
            "SELECT consumed_at, reserved_job_id FROM turn_interrupt_snapshots "
            "WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
    assert row["consumed_at"] is None
    assert row["reserved_job_id"] is None


async def test_stop_between_runtime_and_effect_creates_resume_snapshot(
    tmp_path: Path,
) -> None:
    repository, lease, claim = _claimed_run(tmp_path)

    class _RaceDispatcher:
        def create_live_progress(self, **_: object) -> None:
            return None

        async def dispatch(self, **_: object):  # type: ignore[no-untyped-def]
            repository.request_interrupt(
                InterruptCommand(
                    event_id="stop-race",
                    message_id="stop-race-message",
                    session_key="fake:1",
                    channel="fake",
                    chat_id="1",
                    requested_at=NOW + timedelta(seconds=1),
                )
            )
            return DeliveryResult(
                DeliveryOutcome.CANCELLED,
                EffectRecord(
                    operation_id="cancelled",
                    run_id=claim.run_id,
                    session_key="fake:1",
                    channel="fake",
                    chat_id="1",
                    payload_json='{"text":""}',
                    payload_hash="",
                    provider_uuid="",
                    expected_activity_version=0,
                    state="cancelled",
                    owner_id=lease.owner_id,
                    fencing_epoch=lease.epoch,
                    message_id=None,
                    first_requested_at=None,
                    cancel_on_activity=False,
                ),
            )

    with pytest.raises(TurnInterrupted):
        await RuntimeJobExecutor(
            repository,
            AgentRuntime(_Provider(), tools=ToolRegistry()),
            final_response_dispatcher=_RaceDispatcher(),  # type: ignore[arg-type]
            clock=lambda: NOW + timedelta(seconds=1),
        ).execute(
            claim=claim,
            lease=lease,
            turn=TurnInput(session_key="fake:1", content="原任务"),
            now=NOW,
        )

    with connect_database(repository.database) as connection:
        interrupt = connection.execute(
            "SELECT state FROM session_interrupts WHERE target_run_id = ?",
            (claim.run_id,),
        ).fetchone()
        snapshot = connection.execute(
            "SELECT snapshot_id FROM turn_interrupt_snapshots WHERE source_run_id = ?",
            (claim.run_id,),
        ).fetchone()
    assert interrupt["state"] == "acknowledged"
    assert snapshot is not None


async def test_stop_after_confirmed_effect_does_not_create_duplicate_resume(
    tmp_path: Path,
) -> None:
    repository, lease, claim = _claimed_run(tmp_path)

    class _ConfirmedRaceDispatcher:
        def create_live_progress(self, **_: object) -> None:
            return None

        async def dispatch(self, **_: object):  # type: ignore[no-untyped-def]
            repository.request_interrupt(
                InterruptCommand(
                    event_id="late-stop",
                    message_id="late-stop-message",
                    session_key="fake:1",
                    channel="fake",
                    chat_id="1",
                    requested_at=NOW + timedelta(seconds=1),
                )
            )
            return DeliveryResult(
                DeliveryOutcome.CONFIRMED,
                EffectRecord(
                    operation_id="confirmed",
                    run_id=claim.run_id,
                    session_key="fake:1",
                    channel="fake",
                    chat_id="1",
                    payload_json='{"text":"完成"}',
                    payload_hash="",
                    provider_uuid="",
                    expected_activity_version=0,
                    state="confirmed",
                    owner_id=lease.owner_id,
                    fencing_epoch=lease.epoch,
                    message_id="om-confirmed",
                    first_requested_at=NOW.isoformat(),
                    cancel_on_activity=False,
                ),
            )

    await RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), tools=ToolRegistry()),
        final_response_dispatcher=_ConfirmedRaceDispatcher(),  # type: ignore[arg-type]
        clock=lambda: NOW + timedelta(seconds=1),
    ).execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key="fake:1", content="原任务"),
        now=NOW,
    )

    assert repository.get_job(claim.job_id).state == "succeeded"  # type: ignore[union-attr]
    with connect_database(repository.database) as connection:
        interrupt = connection.execute(
            "SELECT state FROM session_interrupts WHERE target_run_id = ?",
            (claim.run_id,),
        ).fetchone()
        snapshots = connection.execute(
            "SELECT COUNT(*) AS count FROM turn_interrupt_snapshots "
            "WHERE source_run_id = ?",
            (claim.run_id,),
        ).fetchone()["count"]
    assert interrupt["state"] == "acknowledged"
    assert snapshots == 0


class _FailingProvider(ChatProvider):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        raise TimeoutError("provider timeout")
