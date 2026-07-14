from __future__ import annotations

from pathlib import Path

import pytest

from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.memory.optimizer import MemoryOptimizer


class _OptimizerModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def optimize(self, memory: str, self_text: str, pending: str) -> tuple[str, str]:
        if self.fail:
            raise RuntimeError("模型异常")
        assert "旧事实" in pending
        return ("# 长期记忆\n\n- 已合并旧事实", "# MemoPilot\n\n稳定的自我认知")


@pytest.mark.asyncio
async def test_optimizer_snapshot_preserves_new_appends_and_commits(tmp_path: Path) -> None:
    markdown = MarkdownMemoryStore(tmp_path)
    markdown.replace("MEMORY.md", "# 长期记忆")
    markdown.replace("SELF.md", "# MemoPilot")
    markdown.replace("PENDING.md", "- 旧事实")
    optimizer = MemoryOptimizer(markdown, _OptimizerModel())

    async def append_while_optimizing() -> None:
        markdown.append("PENDING.md", "- 新事实")

    result = await optimizer.run(after_snapshot=append_while_optimizing)

    assert result is True
    assert "已合并旧事实" in markdown.read("MEMORY.md")
    assert markdown.read("PENDING.md").strip() == "- 新事实"
    assert not markdown.snapshot_path.exists()


@pytest.mark.asyncio
async def test_optimizer_rolls_back_snapshot_on_failure_or_restart(tmp_path: Path) -> None:
    markdown = MarkdownMemoryStore(tmp_path)
    markdown.replace("PENDING.md", "- 旧事实")
    optimizer = MemoryOptimizer(markdown, _OptimizerModel(fail=True))

    with pytest.raises(RuntimeError, match="模型异常"):
        await optimizer.run()
    assert markdown.read("PENDING.md").strip() == "- 旧事实"
    assert not markdown.snapshot_path.exists()

    markdown.begin_pending_snapshot()
    markdown.append("PENDING.md", "- 重启后新事实")
    assert optimizer.recover() is True
    pending = markdown.read("PENDING.md")
    assert "旧事实" in pending and "重启后新事实" in pending
