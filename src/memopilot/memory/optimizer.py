"""基于 PENDING 快照的长期记忆优化器。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from memopilot.memory.markdown import MarkdownMemoryStore


class OptimizerModel(Protocol):
    async def optimize(self, memory: str, self_text: str, pending: str) -> tuple[str, str]: ...


class MemoryOptimizer:
    def __init__(self, markdown: MarkdownMemoryStore, model: OptimizerModel) -> None:
        self.markdown = markdown
        self.model = model

    async def run(
        self,
        *,
        after_snapshot: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        if self.markdown.snapshot_path.exists():
            raise RuntimeError("检测到未恢复的 PENDING 快照")
        pending = self.markdown.begin_pending_snapshot()
        if not pending.strip():
            self.markdown.commit_pending_snapshot()
            return False
        try:
            if after_snapshot is not None:
                await after_snapshot()
            memory, self_text = await self.model.optimize(
                self.markdown.read("MEMORY.md"),
                self.markdown.read("SELF.md"),
                pending,
            )
            if not memory.strip() or not self_text.strip():
                raise ValueError("优化器不得提交空 MEMORY.md 或 SELF.md")
            self.markdown.replace("MEMORY.md", memory)
            self.markdown.replace("SELF.md", self_text)
            self.markdown.commit_pending_snapshot()
            return True
        except BaseException:
            self.markdown.rollback_pending_snapshot()
            raise

    def recover(self) -> bool:
        return self.markdown.rollback_pending_snapshot()


__all__ = ["MemoryOptimizer"]
