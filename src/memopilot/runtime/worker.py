"""已认领 Job 与 AgentRuntime 之间的最小执行边界。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from memopilot.runtime.engine import AgentRuntime, TurnInput, TurnResult
from memopilot.runtime.persistence import OperationalStepSink
from memopilot.tasks.operational import (
    FenceToken,
    LostLeaseError,
    OperationalRepository,
    RunClaim,
)


class RuntimeJobExecutor:
    def __init__(
        self,
        repository: OperationalRepository,
        runtime: AgentRuntime,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._runtime = runtime
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        *,
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
    ) -> TurnResult:
        if (
            claim.run_id == ""
            or claim.session_key != lease.session_key
            or claim.owner_id != lease.owner_id
            or claim.fencing_epoch != lease.epoch
        ):
            raise ValueError("RunClaim 与 lease 不匹配")
        if turn.session_key != claim.session_key:
            raise ValueError("TurnInput 与 RunClaim 的会话不匹配")
        step_sink = OperationalStepSink(
            self._repository,
            run_id=claim.run_id,
            lease=lease,
            clock=self._clock,
        )
        try:
            result = await self._runtime.run(turn, step_sink=step_sink)
        except Exception:
            try:
                self._repository.finish_job(
                    claim.run_id,
                    lease=lease,
                    outcome="failed",
                    now=now,
                )
            except LostLeaseError:
                pass
            raise
        outcome = "failed" if result.react.infrastructure_error else "succeeded"
        self._repository.finish_job(
            claim.run_id,
            lease=lease,
            outcome=outcome,
            now=now,
        )
        return result


__all__ = ["RuntimeJobExecutor"]
