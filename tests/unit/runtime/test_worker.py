from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.runtime.engine import TurnInput, TurnResult
from memopilot.runtime.react import ReActResult
from memopilot.runtime.worker import RuntimeJobExecutor

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
