from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.tasks.operational import LostLeaseError


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
    history = markdown.read("HISTORY.md")
    assert "[memory_optimizer] PENDING 归档" in history
    assert "- 旧事实" in history
    assert "- 新事实" not in history
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


@pytest.mark.asyncio
async def test_optimizer_does_not_replace_files_after_losing_lease(tmp_path: Path) -> None:
    markdown = MarkdownMemoryStore(tmp_path)
    markdown.replace("MEMORY.md", "# 原记忆")
    markdown.replace("SELF.md", "# 原自我")
    markdown.replace("PENDING.md", "- 旧事实")
    optimizer = MemoryOptimizer(markdown, _OptimizerModel())
    scopes = 0

    @contextmanager
    def fenced_write() -> Iterator[None]:
        nonlocal scopes
        scopes += 1
        if scopes >= 2:
            raise LostLeaseError("模拟 guard 成功后、发布前失权")
        yield

    with pytest.raises(LostLeaseError, match="失权"):
        await optimizer.run(
            assert_current=lambda: None,
            fenced_write=fenced_write,
        )

    assert markdown.read("MEMORY.md").strip() == "# 原记忆"
    assert markdown.read("SELF.md").strip() == "# 原自我"
    assert markdown.snapshot_path.exists()
    assert optimizer.recover() is True
    assert markdown.read("PENDING.md").strip() == "- 旧事实"


def test_optimizer_recovers_both_files_after_process_crash_mid_publish(tmp_path: Path) -> None:
    markdown = MarkdownMemoryStore(tmp_path)
    markdown.replace("MEMORY.md", "# 原记忆")
    markdown.replace("SELF.md", "# 原自我")
    markdown.replace("PENDING.md", "- 旧事实")
    pending = markdown.begin_pending_snapshot()
    assert "旧事实" in pending
    markdown.begin_optimizer_publish(
        memory=markdown.read("MEMORY.md"),
        self_text=markdown.read("SELF.md"),
    )
    markdown.replace("MEMORY.md", "# 新记忆")
    # 模拟进程在第一份正式文件替换后被强杀；新实例只看到磁盘状态。
    restarted = MemoryOptimizer(MarkdownMemoryStore(tmp_path), _OptimizerModel())

    assert restarted.recover() is True
    assert markdown.read("MEMORY.md").strip() == "# 原记忆"
    assert markdown.read("SELF.md").strip() == "# 原自我"
    assert markdown.read("PENDING.md").strip() == "- 旧事实"
    assert not markdown.optimizer_publish_path.exists()
