"""记忆后台任务路由。"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime

from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.post_response import OperationalPostResponseService
from memopilot.memory.vectorization import VectorizationService
from memopilot.tasks.agent_task import AgentTask
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository


class MemoryService:
    def __init__(
        self,
        consolidation: ConsolidationService,
        vectorization: VectorizationService,
        optimizer: MemoryOptimizer,
        repository: OperationalRepository,
        post_response: OperationalPostResponseService | None = None,
    ) -> None:
        self.consolidation = consolidation
        self.vectorization = vectorization
        self.optimizer = optimizer
        self.repository = repository
        self.post_response = post_response

    async def execute_task(
        self, task: AgentTask, *, lease: SessionLease, now: datetime
    ) -> tuple[AgentTask, ...]:
        del now
        kind, session_key, payload = task.kind, task.session_key, task.payload
        def assert_current() -> None:
            self.repository.assert_current_fence(lease)

        def fenced_write() -> AbstractContextManager[None]:
            return self.repository.fenced_write(lease)

        assert_current()
        if kind == "memory.consolidate":
            result = await self.consolidation.run(
                session_key,
                assert_current=assert_current,
                lease=lease,
                fenced_write=fenced_write,
            )
            if result is not None:
                await self.vectorization.run(
                    result.consolidation_id,
                    assert_current=assert_current,
                    lease=lease,
                    fenced_write=fenced_write,
                )
            return ()
        if kind == "memory.vectorize":
            consolidation_id = str(payload.get("consolidation_id") or "")
            if not consolidation_id:
                raise ValueError("memory.vectorize 缺少 consolidation_id")
            await self.vectorization.run(
                consolidation_id,
                assert_current=assert_current,
                lease=lease,
                fenced_write=fenced_write,
            )
            return ()
        if kind == "memory.optimize":
            await self.optimizer.run(
                assert_current=assert_current,
                fenced_write=fenced_write,
            )
            return ()
        if kind == "memory.post_response":
            if self.post_response is None:
                raise RuntimeError("memory.post_response 未装配")
            turn_id = str(payload.get("turn_id") or "")
            if not turn_id:
                raise ValueError("memory.post_response 缺少 turn_id")
            raw_protected_ids = payload.get("protected_ids")
            protected_ids = (
                {str(value) for value in raw_protected_ids}
                if isinstance(raw_protected_ids, list)
                else set()
            )
            await self.post_response.run(
                turn_id=turn_id,
                session_key=session_key,
                protected_ids=protected_ids,
                assert_current=assert_current,
                fenced_write=fenced_write,
            )
            return ()
        if kind == "memory.reinforce":
            usage_ref = str(payload.get("usage_ref") or "")
            raw_ids = payload.get("item_ids")
            item_ids = tuple(str(value) for value in raw_ids) if isinstance(raw_ids, list) else ()
            if not usage_ref or not item_ids:
                raise ValueError("memory.reinforce 缺少 usage_ref 或 item_ids")
            assert_current()
            with fenced_write():
                self.vectorization.store.reinforce_items_once(
                    item_ids,
                    usage_ref=usage_ref,
                )
            return ()
        raise ValueError(f"不支持的记忆任务: {kind}")


__all__ = ["MemoryService"]
