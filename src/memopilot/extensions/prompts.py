"""可预算、可按运行场景筛选的 Prompt 扩展合同。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptSectionRender:
    """插件已经渲染好的 Prompt section。"""

    name: str
    content: str
    is_static: bool
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Prompt section name 不能为空")


@dataclass(slots=True)
class PromptRenderContext:
    """Prompt 插件每轮可读取并改写的上下文。"""

    session_key: str
    content: str
    scope: str
    system_prompt: str
    history: tuple[Any, ...] = ()
    channel: str = ""
    chat_id: str = ""
    media: tuple[str, ...] = ()
    extra_hints: list[str] = field(default_factory=list)
    system_sections_top: list[PromptSectionRender] = field(default_factory=list)
    system_sections_bottom: list[PromptSectionRender] = field(default_factory=list)


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
    def __init__(
        self,
        blocks: tuple[PromptBlock, ...] = (),
        *,
        static_cache_max_entries: int = 128,
    ) -> None:
        if static_cache_max_entries <= 0:
            raise ValueError("static Prompt 缓存上限必须大于 0")
        seen: set[str] = set()
        for block in blocks:
            if block.block_id in seen:
                raise ValueError(f"PromptBlock 重复: {block.block_id}")
            seen.add(block.block_id)
        self._blocks = tuple(sorted(blocks, key=lambda item: (item.priority, item.block_id)))
        self._static_cache_max_entries = static_cache_max_entries
        self._static_sections: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    @property
    def static_cache_size(self) -> int:
        return len(self._static_sections)

    def prepare_sections(
        self,
        *,
        scope: str,
        sections: tuple[PromptSectionRender, ...],
    ) -> tuple[PromptSectionRender, ...]:
        prepared: list[PromptSectionRender] = []
        for section in sections:
            if not section.is_static:
                prepared.append(replace(section, cache_hit=False))
                continue
            key = (scope, section.name, section.content)
            cached = self._static_sections.get(key)
            if cached is None:
                self._static_sections[key] = section.content
                self._static_sections.move_to_end(key)
                while len(self._static_sections) > self._static_cache_max_entries:
                    self._static_sections.popitem(last=False)
                prepared.append(replace(section, cache_hit=False))
            else:
                self._static_sections.move_to_end(key)
                prepared.append(replace(section, content=cached, cache_hit=True))
        return tuple(prepared)

    def render(
        self,
        *,
        scope: str,
        base_prompt: str = "",
        max_chars: int,
        top_sections: tuple[PromptSectionRender, ...] = (),
        bottom_sections: tuple[PromptSectionRender, ...] = (),
    ) -> str:
        if max_chars <= 0:
            raise ValueError("Prompt 总预算必须大于 0")
        core = base_prompt.strip()
        prepared_top = self.prepare_sections(scope=scope, sections=top_sections)
        prepared_bottom = self.prepare_sections(scope=scope, sections=bottom_sections)
        if len(core) >= max_chars:
            return core[:max_chars]

        candidates: list[tuple[bool, str]] = [
            (True, section.content) for section in prepared_top
        ]
        for block in self._blocks:
            if scope not in block.scopes and "all" not in block.scopes:
                continue
            candidates.append((False, block.content.strip()[: block.max_chars]))
        candidates.extend((False, section.content) for section in prepared_bottom)

        if not core:
            result = ""
            for _, content in candidates:
                result = _append_with_budget(result, content, max_chars)
            return result

        remaining = max_chars - len(core)
        selected_top: list[str] = []
        selected_tail: list[str] = []
        for is_top, content in candidates:
            text = content.strip()
            if not text or remaining <= 2:
                continue
            selected = text[: remaining - 2]
            if not selected:
                continue
            (selected_top if is_top else selected_tail).append(selected)
            remaining -= len(selected) + 2
        return "\n\n".join((*selected_top, core, *selected_tail))


def _append_with_budget(current: str, content: str, max_chars: int) -> str:
    text = content.strip()
    if not text or len(current) >= max_chars:
        return current
    separator = "\n\n" if current else ""
    remaining = max_chars - len(current) - len(separator)
    if remaining <= 0:
        return current
    return current + separator + text[:remaining]


__all__ = [
    "PromptBlock",
    "PromptRenderContext",
    "PromptRenderer",
    "PromptSectionRender",
]
