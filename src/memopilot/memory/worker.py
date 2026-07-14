"""记忆后台任务路由。"""

from __future__ import annotations

from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.vectorization import VectorizationService


class MemoryJobRouter:
    def __init__(
        self,
        consolidation: ConsolidationService,
        vectorization: VectorizationService,
        optimizer: MemoryOptimizer,
    ) -> None:
        self.consolidation = consolidation
        self.vectorization = vectorization
        self.optimizer = optimizer

    async def execute(
        self, *, kind: str, session_key: str, payload: dict[str, object]
    ) -> None:
        if kind == "memory.consolidate":
            await self.consolidation.run(session_key)
            return
        if kind == "memory.vectorize":
            consolidation_id = str(payload.get("consolidation_id") or "")
            if not consolidation_id:
                raise ValueError("memory.vectorize 缺少 consolidation_id")
            await self.vectorization.run(consolidation_id)
            return
        if kind == "memory.optimize":
            await self.optimizer.run()
            return
        raise ValueError(f"不支持的记忆任务: {kind}")


__all__ = ["MemoryJobRouter"]
