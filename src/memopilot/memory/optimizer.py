"""将 PENDING 记忆优化并发布到长期记忆文件。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext
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
        assert_current: Callable[[], None] | None = None,
        fenced_write: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> bool:
        guard = assert_current or (lambda: None)
        write_scope = fenced_write or nullcontext
        guard()
        if self.markdown.optimizer_publish_path.exists():
            with write_scope():
                self.markdown.recover_optimizer_publish()
        elif self.markdown.snapshot_path.exists():
            with write_scope():
                self.markdown.rollback_pending_snapshot()
        with write_scope():
            pending = self.markdown.begin_pending_snapshot()
        if not pending.strip():
            with write_scope():
                self.markdown.commit_pending_snapshot()
            return False
        original_memory = self.markdown.read("MEMORY.md")
        original_self = self.markdown.read("SELF.md")
        original_history = self.markdown.read("HISTORY.md")
        try:
            if after_snapshot is not None:
                await after_snapshot()
            memory, self_text = await self.model.optimize(
                self.markdown.read("MEMORY.md"),
                self.markdown.read("SELF.md"),
                pending,
            )
            guard()
            if not memory.strip() or not self_text.strip():
                raise ValueError("优化结果必须同时包含 MEMORY.md 和 SELF.md")
            with write_scope():
                try:
                    self.markdown.begin_optimizer_publish(
                        memory=original_memory,
                        self_text=original_self,
                        history=original_history,
                    )
                    self.markdown.replace("MEMORY.md", memory)
                    self.markdown.replace("SELF.md", self_text)
                    self.markdown.append(
                        "HISTORY.md",
                        f"[memory_optimizer] PENDING 内容：\n{pending.strip()}",
                    )
                    self.markdown.mark_optimizer_publish_committed()
                    self.markdown.recover_optimizer_publish()
                except BaseException:
                    self.markdown.recover_optimizer_publish()
                    raise
            return True
        except BaseException:
            with write_scope():
                self.markdown.rollback_pending_snapshot()
            raise

    def recover(self) -> bool:
        return (
            self.markdown.recover_optimizer_publish() or self.markdown.rollback_pending_snapshot()
        )


__all__ = ["MemoryOptimizer"]
