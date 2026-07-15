"""可预算、可按运行场景筛选的 PromptBlock。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptBlock:
    block_id: str
    content: str
    priority: int = 100
    scopes: tuple[str, ...] = ("all",)
    max_chars: int = 1200

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("PromptBlock block_id 不能为空")
        if not self.scopes or any(not scope.strip() for scope in self.scopes):
            raise ValueError(f"PromptBlock {self.block_id} scopes 不能为空")
        if self.max_chars <= 0:
            raise ValueError(f"PromptBlock {self.block_id} max_chars 必须大于 0")


class PromptRenderer:
    def __init__(self, blocks: tuple[PromptBlock, ...] = ()) -> None:
        seen: set[str] = set()
        for block in blocks:
            if block.block_id in seen:
                raise ValueError(f"PromptBlock 重复: {block.block_id}")
            seen.add(block.block_id)
        self._blocks = tuple(sorted(blocks, key=lambda item: (item.priority, item.block_id)))

    def render(self, *, scope: str, base_prompt: str = "", max_chars: int) -> str:
        if max_chars <= 0:
            raise ValueError("Prompt 总预算必须大于 0")
        result = base_prompt.strip()[:max_chars]
        for block in self._blocks:
            if scope not in block.scopes and "all" not in block.scopes:
                continue
            content = block.content.strip()[: block.max_chars]
            if not content:
                continue
            separator = "\n\n" if result else ""
            remaining = max_chars - len(result) - len(separator)
            if remaining <= 0:
                break
            result += separator + content[:remaining]
        return result


__all__ = ["PromptBlock", "PromptRenderer"]

