"""记忆后台任务的统一处理服务。"""

from __future__ import annotations

from datetime import datetime

from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.post_response import OperationalPostResponseService
from memopilot.memory.vectorization import VectorizationService
from memopilot.persistence.conversation import ConversationRepository
from memopilot.tasks.agent_task import AgentTask


class MemoryService:
    def __init__(
        self,
        consolidation: ConsolidationService,
        vectorization: VectorizationService,
        optimizer: MemoryOptimizer,
        repository: ConversationRepository,
        post_response: OperationalPostResponseService | None = None,
    ) -> None:
        self.consolidation = consolidation
        self.vectorization = vectorization
        self.optimizer = optimizer
        self.repository = repository
        self.post_response = post_response

    async def execute_task(self, task: AgentTask, *, now: datetime) -> tuple[AgentTask, ...]:
        del now
        kind, session_key, payload = task.kind, task.session_key, task.payload
        if kind == "memory.consolidate":
            result = await self.consolidation.run(session_key)
            if result is not None:
                await self.vectorization.run(result.consolidation_id)
            return ()
        if kind == "memory.vectorize":
            consolidation_id = str(payload.get("consolidation_id") or "")
            if not consolidation_id:
                raise ValueError("memory.vectorize 缺少 consolidation_id")
            await self.vectorization.run(consolidation_id)
            return ()
        if kind == "memory.optimize":
            await self.optimizer.run()
            return ()
        if kind == "memory.post_response":
            if self.post_response is None:
                raise RuntimeError("memory.post_response 未配置处理器")
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
            )
            return ()
        if kind == "memory.reinforce":
            usage_ref = str(payload.get("usage_ref") or "")
            raw_ids = payload.get("item_ids")
            item_ids = tuple(str(value) for value in raw_ids) if isinstance(raw_ids, list) else ()
            if not usage_ref or not item_ids:
                raise ValueError("memory.reinforce 缺少 usage_ref 或 item_ids")
            self.vectorization.store.reinforce_items_once(item_ids, usage_ref=usage_ref)
            return ()
        raise ValueError(f"不支持的记忆任务: {kind}")


__all__ = ["MemoryService"]
