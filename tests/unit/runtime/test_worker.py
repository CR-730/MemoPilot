from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.runtime.engine import TurnInput, TurnResult
from memopilot.runtime.react import ReActResult
from memopilot.runtime.worker import (
    RuntimeJobExecutor,
    ScheduleJobRequeued,
    SystemJobResult,
)

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _Repository:
    def __init__(self) -> None:
        self.commits: list[dict[str, Any]] = []
        self.finishes: list[str] = []

    def commit_successful_turn(self, run_id: str, **kwargs: Any) -> None:
        self.commits.append({"run_id": run_id, **kwargs})

    def finish_job(self, run_id: str, *, outcome: str, **kwargs: Any) -> None:
        self.finishes.append(outcome)

    def get_job(self, job_id: str):
        return SimpleNamespace(job_id=job_id, kind="agent.turn", payload_json="{}")


class _Runtime:
    async def run(self, turn: TurnInput, **kwargs: Any) -> TurnResult:
        react = ReActResult(
            reply="记忆回复",
            messages=(),
            iterations=1,
            tool_chain=(),
            exit_reason="completed",
        )
        return TurnResult("记忆回复", (), react, (), ())


@pytest.mark.asyncio
async def test_successful_runtime_atomically_commits_turn_instead_of_plain_finish() -> None:
    repository = _Repository()
    executor = RuntimeJobExecutor(repository, _Runtime())  # type: ignore[arg-type]
    claim = SimpleNamespace(
        job_id="job-1",
        run_id="run-1",
        session_key="feishu:chat-1",
        owner_id="worker-1",
        fencing_epoch=1,
    )
    lease = SimpleNamespace(
        session_key="feishu:chat-1",
        owner_id="worker-1",
        epoch=1,
    )

    await executor.execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key="feishu:chat-1", content="用户问题"),
        now=NOW,
    )

    assert repository.finishes == []
    assert len(repository.commits) == 1
    assert repository.commits[0]["run_id"] == "run-1"
    assert repository.commits[0]["user_content"] == "用户问题"
    assert repository.commits[0]["assistant_content"] == "记忆回复"


class _MemoryRepository(_Repository):
    def get_job(self, job_id: str):
        return SimpleNamespace(
            job_id=job_id,
            kind="memory.vectorize",
            payload_json='{"consolidation_id":"con-1"}',
        )


class _MemoryJobs:
    def __init__(self) -> None:
        self.calls = []

    async def execute(
        self,
        *,
        kind: str,
        session_key: str,
        payload: dict[str, Any],
        run_id: str,
        lease: Any,
    ) -> None:
        self.calls.append((kind, session_key, payload, run_id, lease))


@pytest.mark.asyncio
async def test_memory_job_bypasses_agent_runtime_and_finishes_normally() -> None:
    repository = _MemoryRepository()
    memory_jobs = _MemoryJobs()
    executor = RuntimeJobExecutor(
        repository,
        _Runtime(),
        memory_jobs=memory_jobs,  # type: ignore[arg-type]
    )
    claim = SimpleNamespace(
        job_id="job-memory",
        run_id="run-memory",
        session_key="feishu:chat-1",
        owner_id="worker-1",
        fencing_epoch=1,
    )
    lease = SimpleNamespace(session_key="feishu:chat-1", owner_id="worker-1", epoch=1)

    result = await executor.execute(
        claim=claim,
        lease=lease,
        turn=TurnInput(session_key="feishu:chat-1", content=""),
        now=NOW,
    )

    assert result is None
    assert memory_jobs.calls == [
        (
            "memory.vectorize",
            "feishu:chat-1",
            {"consolidation_id": "con-1"},
            "run-memory",
            lease,
        )
    ]
    assert repository.commits == []
    assert repository.finishes == ["succeeded"]


class _SystemRepository(_Repository):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.fence_checks = 0

    def get_job(self, job_id: str):
        return SimpleNamespace(
            job_id=job_id,
            kind=self.kind,
            payload_json='{"value":1}',
            activity_version=1,
        )

    def assert_current_fence(self, lease: Any) -> None:
        self.fence_checks += 1


class _SystemJobs:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, *, kind: str, **kwargs: Any) -> SystemJobResult:
        self.calls.append(kind)
        return SystemJobResult(outcome="succeeded")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["proactive.tick", "schedule.run", "drift.run"])
async def test_system_jobs_bypass_passive_turn_commit(kind: str) -> None:
    repository = _SystemRepository(kind)
    system_jobs = _SystemJobs()
    executor = RuntimeJobExecutor(
        repository,  # type: ignore[arg-type]
        _Runtime(),
        system_jobs=system_jobs,
    )
    claim = SimpleNamespace(
        job_id="job-system",
        run_id="run-system",
        session_key="feishu:chat-1",
        owner_id="worker-1",
        fencing_epoch=1,
    )
    lease = SimpleNamespace(session_key="feishu:chat-1", owner_id="worker-1", epoch=1)

    result = await executor.execute(
        claim=claim,
        lease=lease,
        turn=TurnInput("feishu:chat-1", "", received_at=NOW),
        now=NOW,
    )

    assert result is None
    assert system_jobs.calls == [kind]
    assert repository.commits == []
    assert repository.finishes == ["succeeded"]
    assert repository.fence_checks >= 1


@pytest.mark.asyncio
async def test_schedule_cancelled_by_new_activity_is_requeued_not_finished() -> None:
    repository = _SystemRepository("schedule.run")
    repository.requeued = False
    repository.get_activity_version = lambda session_key: 2

    def requeue(*args: Any, **kwargs: Any) -> str:
        repository.requeued = True
        return "requeued"

    repository.requeue_preempted_schedule = requeue

    class CancelledSystemJobs(_SystemJobs):
        async def execute(self, *, kind: str, **kwargs: Any) -> SystemJobResult:
            return SystemJobResult(outcome="cancelled")

    executor = RuntimeJobExecutor(
        repository,  # type: ignore[arg-type]
        _Runtime(),
        system_jobs=CancelledSystemJobs(),
    )
    claim = SimpleNamespace(
        job_id="job-system",
        run_id="run-system",
        session_key="feishu:chat-1",
        owner_id="worker-1",
        fencing_epoch=1,
    )
    lease = SimpleNamespace(session_key="feishu:chat-1", owner_id="worker-1", epoch=1)

    with pytest.raises(ScheduleJobRequeued):
        await executor.execute(
            claim=claim,
            lease=lease,
            turn=TurnInput("feishu:chat-1", "", received_at=NOW),
            now=NOW,
        )

    assert repository.requeued is True
    assert repository.finishes == []
