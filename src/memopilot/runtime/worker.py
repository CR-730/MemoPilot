"""已认领 Job 与 AgentRuntime 之间的最小执行边界。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol

from memopilot.delivery.feishu import DeliveryOutcome, FinalResponseDispatcher
from memopilot.runtime.engine import AgentRuntime, TurnInput, TurnResult
from memopilot.runtime.interrupts import CompositeProgressObserver, InterruptProgressRecorder
from memopilot.runtime.persistence import OperationalStepSink
from memopilot.tasks.operational import (
    FenceToken,
    LostLeaseError,
    OperationalRepository,
    PendingInterruptError,
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


class TurnInterrupted(RuntimeError):
    """当前 Run 已按用户 `/stop` 请求安全中断并保存快照。"""


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
        turn = replace(turn, memory_source_ref=f"run:{claim.run_id}")
        step_sink = OperationalStepSink(
            self._repository,
            run_id=claim.run_id,
            lease=lease,
            clock=self._clock,
        )
        live_progress = None
        if self._final_response_dispatcher is not None:
            try:
                live_progress = self._final_response_dispatcher.create_live_progress(
                    claim=claim,
                    lease=lease,
                )
            except Exception as exc:
                logger.warning("飞书 live 初始化异常，核心 Runtime 继续执行: %s", exc)
        recorder = InterruptProgressRecorder()
        progress = CompositeProgressObserver(
            (recorder,) if live_progress is None else (recorder, live_progress)
        )
        try:
            result = await self._runtime.run(
                turn,
                step_sink=step_sink,
                progress=progress,
                memory_assert_current=lambda: self._repository.assert_current_fence(lease),
                memory_fenced_write=lambda: self._repository.fenced_write(lease),
            )
        except asyncio.CancelledError:
            if self._repository.has_pending_interrupt(claim.run_id):
                self._repository.finish_interrupted_run(
                    claim.run_id,
                    lease=lease,
                    snapshot=recorder.snapshot(
                        original_message=turn.interrupt_original_message or turn.content
                    ),
                    now=self._clock(),
                )
                raise TurnInterrupted(claim.run_id) from None
            raise
        except Exception:
            try:
                self._repository.finish_job(
                    claim.run_id,
                    lease=lease,
                    outcome="failed",
                    now=self._clock(),
                    resume_snapshot_id=turn.resume_snapshot_id,
                )
            except LostLeaseError:
                pass
            raise
        if live_progress is not None:
            try:
                await live_progress.finalize()
            except asyncio.CancelledError:
                if self._repository.has_pending_interrupt(claim.run_id):
                    self._repository.finish_interrupted_run(
                        claim.run_id,
                        lease=lease,
                        snapshot=recorder.snapshot(
                            original_message=turn.interrupt_original_message or turn.content
                        ),
                        now=self._clock(),
                    )
                    raise TurnInterrupted(claim.run_id) from None
                raise
            except Exception as exc:
                logger.warning("飞书过程卡定格异常，最终回复继续发送: %s", exc)
        delivery_confirmed = False
        if result.react.infrastructure_error:
            outcome = "failed"
        elif self._final_response_dispatcher is None:
            outcome = "succeeded"
        else:
            try:
                delivery = await self._final_response_dispatcher.dispatch(
                    claim=claim,
                    lease=lease,
                    text=result.reply,
                )
            except asyncio.CancelledError:
                if self._repository.has_pending_interrupt(claim.run_id):
                    self._repository.finish_interrupted_run(
                        claim.run_id,
                        lease=lease,
                        snapshot=recorder.snapshot(
                            original_message=turn.interrupt_original_message or turn.content
                        ),
                        now=self._clock(),
                    )
                    raise TurnInterrupted(claim.run_id) from None
                raise
            outcome = {
                DeliveryOutcome.CONFIRMED: "succeeded",
                DeliveryOutcome.CANCELLED: "cancelled",
                DeliveryOutcome.FAILED: "failed",
                DeliveryOutcome.NEEDS_REVIEW: "needs_review",
            }[delivery.outcome]
            delivery_confirmed = delivery.outcome is DeliveryOutcome.CONFIRMED
        try:
            if outcome == "succeeded":
                self._repository.commit_successful_turn(
                    claim.run_id,
                    lease=lease,
                    user_content=turn.current_user_content or turn.content,
                    assistant_content=result.reply,
                    cited_memory_ids=result.cited_memory_ids,
                    now=self._clock(),
                    resume_snapshot_id=turn.resume_snapshot_id,
                    reject_pending_interrupt=not delivery_confirmed,
                    acknowledge_pending_interrupt=delivery_confirmed,
                )
            else:
                self._repository.finish_job(
                    claim.run_id,
                    lease=lease,
                    outcome=outcome,
                    now=self._clock(),
                    resume_snapshot_id=turn.resume_snapshot_id,
                    reject_pending_interrupt=not delivery_confirmed,
                    acknowledge_pending_interrupt=delivery_confirmed,
                )
        except PendingInterruptError:
            self._repository.finish_interrupted_run(
                claim.run_id,
                lease=lease,
                snapshot=recorder.snapshot(
                    original_message=turn.interrupt_original_message or turn.content
                ),
                now=self._clock(),
            )
            raise TurnInterrupted(claim.run_id) from None
        return result


__all__ = ["MemoryJobExecutor", "RuntimeJobExecutor", "TurnInterrupted"]
