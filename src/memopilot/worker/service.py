"""Redis Job 到 Agent Runtime 的 Worker 纵向执行器。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from memopilot.runtime.engine import TurnInput
from memopilot.runtime.interrupts import render_resumed_message
from memopilot.runtime.worker import RuntimeJobExecutor, TurnInterrupted
from memopilot.tasks.interrupts import InterruptSignalPort
from memopilot.tasks.lease import SessionLease, SessionLeaseManager
from memopilot.tasks.operational import LostLeaseError, OperationalRepository, RunClaim
from memopilot.tasks.recovery import (
    PendingDisposition,
    PendingMessageReclaimer,
)
from memopilot.tasks.redis_queue import QueueMessage, RedisTaskQueue

_TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled", "needs_review"})


class WorkerService:
    def __init__(
        self,
        repository: OperationalRepository,
        queue: RedisTaskQueue,
        leases: SessionLeaseManager,
        executor: RuntimeJobExecutor,
        *,
        owner_id: str,
        clock: Callable[[], datetime] | None = None,
        heartbeat_interval: float | None = None,
        pending_min_idle: timedelta = timedelta(seconds=60),
        stale_heartbeat: timedelta = timedelta(seconds=60),
        pending_reclaim_every: int = 20,
        interrupt_poll_interval: float = 0.1,
        monotonic: Callable[[], float] = time.monotonic,
        interrupt_signal: InterruptSignalPort | None = None,
    ) -> None:
        self._repository = repository
        self._queue = queue
        self._leases = leases
        self._executor = executor
        self._owner_id = owner_id
        self._clock = clock or (lambda: datetime.now(UTC))
        if pending_min_idle.total_seconds() <= 0 or stale_heartbeat.total_seconds() <= 0:
            raise ValueError("Pending 空闲阈值和心跳过期阈值必须大于 0")
        if pending_reclaim_every < 1:
            raise ValueError("Pending 周期扫描间隔必须至少为 1")
        self._pending_min_idle = pending_min_idle
        self._stale_heartbeat = stale_heartbeat
        self._pending_reclaim_every = pending_reclaim_every
        self._new_reads_since_reclaim = 0
        self._reclaimer = PendingMessageReclaimer(repository, queue, leases)
        derived_interval = max(0.05, leases.ttl_ms / 3000)
        self._heartbeat_interval = heartbeat_interval or derived_interval
        if self._heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval 必须大于 0")
        if interrupt_poll_interval <= 0:
            raise ValueError("interrupt_poll_interval 必须大于 0")
        self._interrupt_poll_interval = interrupt_poll_interval
        self._monotonic = monotonic
        self._interrupt_signal = interrupt_signal

    async def run_once(self) -> bool:
        pending = None
        if self._new_reads_since_reclaim >= self._pending_reclaim_every:
            pending = await self._reclaim_pending()
            self._new_reads_since_reclaim = 0
        if pending is None:
            message = await self._queue.read_next(consumer_id=self._owner_id)
            if message is not None:
                self._new_reads_since_reclaim += 1
        else:
            message = None
        if message is None:
            if pending is None:
                pending = await self._reclaim_pending()
                self._new_reads_since_reclaim = 0
            if pending is None:
                return False
            message, disposition = pending
            if disposition is PendingDisposition.CLEANUP:
                await self._queue.acknowledge(message)
                return True
        else:
            disposition = None
        lease = await self._leases.acquire(
            message.session_key,
            owner_id=self._owner_id,
            now=self._clock(),
        )
        if lease is None:
            return False
        try:
            if disposition is PendingDisposition.RESUME:
                job = self._repository.get_job(message.job_id)
                if job is not None and job.state == "running":
                    recovery = self._repository.recover_stale_job(
                        message.job_id,
                        lease=lease,
                        now=self._clock(),
                        heartbeat_before=self._clock() - self._stale_heartbeat,
                    )
                    if recovery == "needs_review":
                        await self._ack_if_terminal(message)
                        return True
            claim = self._repository.claim_job(
                message.job_id,
                lease=lease,
                now=self._clock(),
            )
            if claim is None:
                await self._ack_if_terminal(message)
                return True
            turn = self._turn_input(message, claim)
            try:
                await self._execute_with_heartbeat(claim, lease, turn)
            except Exception:
                await self._ack_if_terminal(message)
                raise
            await self._ack_if_terminal(message)
            return True
        finally:
            await self._leases.release(lease)

    async def run_forever(self, *, idle_interval: float = 0.05) -> None:
        if idle_interval <= 0:
            raise ValueError("idle_interval 必须大于 0")
        await self._queue.ensure_consumer_groups()
        while True:
            processed = await self.run_once()
            if not processed:
                await asyncio.sleep(idle_interval)

    async def _reclaim_pending(
        self,
    ) -> tuple[QueueMessage, PendingDisposition] | None:
        heartbeat_before = self._clock() - self._stale_heartbeat
        for priority in range(4):
            claimed = await self._reclaimer.reclaim_one(
                priority=priority,
                consumer_id=self._owner_id,
                min_idle=self._pending_min_idle,
                heartbeat_before=heartbeat_before,
            )
            if claimed is not None:
                return claimed.message, claimed.disposition
        return None

    async def _execute_with_heartbeat(
        self,
        claim: RunClaim,
        lease: SessionLease,
        turn: TurnInput,
    ) -> None:
        execution = asyncio.create_task(
            self._executor.execute(
                claim=claim,
                lease=lease,
                turn=turn,
                now=self._clock(),
            )
        )
        next_heartbeat = self._monotonic() + self._heartbeat_interval
        try:
            while True:
                timeout = min(
                    self._interrupt_poll_interval,
                    max(0.0, next_heartbeat - self._monotonic()),
                )
                done, _ = await asyncio.wait(
                    {execution},
                    timeout=timeout,
                )
                if done:
                    await execution
                    return
                redis_pending = False
                if self._interrupt_signal is not None:
                    try:
                        redis_pending = await self._interrupt_signal.pending(claim.run_id)
                    except Exception:
                        redis_pending = False
                if redis_pending or self._repository.has_pending_interrupt(claim.run_id):
                    execution.cancel()
                    try:
                        await execution
                    except TurnInterrupted:
                        pass
                    if self._interrupt_signal is not None:
                        try:
                            await self._interrupt_signal.clear(claim.run_id)
                        except Exception:
                            pass
                    return
                if self._monotonic() < next_heartbeat:
                    continue
                if not await self._leases.renew(lease, now=self._clock()):
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    raise LostLeaseError("会话 Lease 续租失败，当前 Worker 已停止提交")
                self._repository.heartbeat_run(
                    claim.run_id,
                    lease=lease,
                    now=self._clock(),
                )
                next_heartbeat = self._monotonic() + self._heartbeat_interval
        except BaseException:
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            raise

    def _turn_input(self, message: QueueMessage, claim: RunClaim) -> TurnInput:
        job = self._repository.get_job(message.job_id)
        if job is None:
            raise KeyError(message.job_id)
        payload = json.loads(job.payload_json)
        content = str(payload.get("text") or "")
        snapshot = self._repository.reserve_interrupt_snapshot(
            claim.session_key,
            job_id=claim.job_id,
            now=self._clock(),
        )
        if snapshot is None:
            return TurnInput(
                session_key=claim.session_key,
                content=content,
                interrupt_original_message=content,
            )
        return TurnInput(
            session_key=claim.session_key,
            content=render_resumed_message(snapshot, content),
            resume_snapshot_id=snapshot.snapshot_id,
            interrupt_original_message=snapshot.original_message,
        )

    async def _ack_if_terminal(self, message: QueueMessage) -> None:
        job = self._repository.get_job(message.job_id)
        if job is not None and job.state in _TERMINAL_JOB_STATES:
            await self._queue.acknowledge(message)


__all__ = ["WorkerService"]
