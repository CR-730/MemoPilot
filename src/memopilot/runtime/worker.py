"""已认领 Job 与 AgentRuntime 之间的最小执行边界。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from memopilot.delivery.feishu import DeliveryOutcome, FinalResponseDispatcher
from memopilot.runtime.engine import AgentRuntime, TurnInput, TurnResult
from memopilot.runtime.persistence import OperationalStepSink
from memopilot.tasks.operational import (
    FenceToken,
    LostLeaseError,
    OperationalRepository,
    RunClaim,
)

logger = logging.getLogger(__name__)


class MemoryJobExecutor(Protocol):
    async def execute(
        self,
        *,
        kind: str,
        session_key: str,
        payload: dict[str, object],
        run_id: str,
        lease: FenceToken,
    ) -> None: ...


class RuntimeJobExecutor:
    def __init__(
        self,
        repository: OperationalRepository,
        runtime: AgentRuntime,
        *,
        final_response_dispatcher: FinalResponseDispatcher | None = None,
        memory_jobs: MemoryJobExecutor | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._runtime = runtime
        self._final_response_dispatcher = final_response_dispatcher
        self._memory_jobs = memory_jobs
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        *,
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
    ) -> TurnResult | None:
        if (
            claim.run_id == ""
            or claim.session_key != lease.session_key
            or claim.owner_id != lease.owner_id
            or claim.fencing_epoch != lease.epoch
        ):
            raise ValueError("RunClaim 与 lease 不匹配")
        job = self._repository.get_job(claim.job_id)
        if job is None:
            raise KeyError(claim.job_id)
        if job.kind.startswith("memory."):
            if self._memory_jobs is None:
                raise RuntimeError(f"未配置记忆任务执行器: {job.kind}")
            payload = json.loads(job.payload_json)
            if not isinstance(payload, dict):
                raise ValueError("记忆任务 payload 必须是 JSON 对象")
            try:
                await self._memory_jobs.execute(
                    kind=job.kind,
                    session_key=claim.session_key,
                    payload={str(key): value for key, value in payload.items()},
                    run_id=claim.run_id,
                    lease=lease,
                )
                self._repository.finish_job(
                    claim.run_id,
                    lease=lease,
                    outcome="succeeded",
                    now=now,
                )
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
            return None
        if turn.session_key != claim.session_key:
            raise ValueError("TurnInput 与 RunClaim 的会话不匹配")
        step_sink = OperationalStepSink(
            self._repository,
            run_id=claim.run_id,
            lease=lease,
            clock=self._clock,
        )
        progress = None
        if self._final_response_dispatcher is not None:
            try:
                progress = self._final_response_dispatcher.create_live_progress(
                    claim=claim,
                    lease=lease,
                )
            except Exception as exc:
                logger.warning("飞书 live 初始化异常，核心 Runtime 继续执行: %s", exc)
        try:
            result = await self._runtime.run(
                turn,
                step_sink=step_sink,
                progress=progress,
            )
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
        if progress is not None:
            try:
                await progress.finalize()
            except Exception as exc:
                logger.warning("飞书过程卡定格异常，最终回复继续发送: %s", exc)
        if result.react.infrastructure_error:
            outcome = "failed"
        elif self._final_response_dispatcher is None:
            outcome = "succeeded"
        else:
            delivery = await self._final_response_dispatcher.dispatch(
                claim=claim,
                lease=lease,
                text=result.reply,
            )
            outcome = {
                DeliveryOutcome.CONFIRMED: "succeeded",
                DeliveryOutcome.CANCELLED: "cancelled",
                DeliveryOutcome.FAILED: "failed",
                DeliveryOutcome.NEEDS_REVIEW: "needs_review",
            }[delivery.outcome]
        if outcome == "succeeded":
            self._repository.commit_successful_turn(
                claim.run_id,
                lease=lease,
                user_content=turn.content,
                assistant_content=result.reply,
                now=now,
            )
        else:
            self._repository.finish_job(
                claim.run_id,
                lease=lease,
                outcome=outcome,
                now=now,
            )
        return result


__all__ = ["MemoryJobExecutor", "RuntimeJobExecutor"]
