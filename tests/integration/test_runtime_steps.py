from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import ToolRegistry
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import (
    InboundCommand,
    LostLeaseError,
    OperationalRepository,
    RunClaim,
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


class _FailingProvider(ChatProvider):
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        raise TimeoutError("provider timeout")
