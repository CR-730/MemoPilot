"""记忆后台任务路由。"""

from __future__ import annotations

from contextlib import AbstractContextManager

from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.vectorization import VectorizationService
from memopilot.tasks.operational import FenceToken, OperationalRepository


class MemoryJobRouter:
    def __init__(
        self,
        consolidation: ConsolidationService,
        vectorization: VectorizationService,
        optimizer: MemoryOptimizer,
        repository: OperationalRepository,
    ) -> None:
        self.consolidation = consolidation
        self.vectorization = vectorization
        self.optimizer = optimizer
        self.repository = repository

    async def execute(
        self,
        *,
        kind: str,
        session_key: str,
        payload: dict[str, object],
        run_id: str,
        lease: FenceToken,
    ) -> None:
        del run_id

        def assert_current() -> None:
            self.repository.assert_current_fence(lease)

        def fenced_write() -> AbstractContextManager[None]:
            return self.repository.fenced_write(lease)

        assert_current()
        if kind == "memory.consolidate":
            await self.consolidation.run(
                session_key,
                assert_current=assert_current,
                lease=lease,
                fenced_write=fenced_write,
            )
            return
        if kind == "memory.vectorize":
            consolidation_id = str(payload.get("consolidation_id") or "")
            if not consolidation_id:
                raise ValueError("memory.vectorize 缺少 consolidation_id")
            await self.vectorization.run(
                consolidation_id,
                assert_current=assert_current,
                fenced_write=fenced_write,
            )
            return
        if kind == "memory.optimize":
            await self.optimizer.run(
                assert_current=assert_current,
                fenced_write=fenced_write,
            )
            return
        raise ValueError(f"不支持的记忆任务: {kind}")


__all__ = ["MemoryJobRouter"]
