from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import AfterReasoningCtx, AfterReasoningInput
from memopilot.extensions.prompts import PromptRenderContext, PromptSectionRender
from memopilot.runtime.react import ReActResult

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_MEME_RE = re.compile(r"<meme:([a-zA-Z0-9_-]+)>", re.IGNORECASE)


class MemeCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.mtime = -1
        self.categories: dict[str, tuple[str, tuple[str, ...]]] = {}
        self.aliases: dict[str, str] = {}

    def load(self) -> None:
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            self.mtime = -1
            self.categories = {}
            self.aliases = {}
            return
        mtime = manifest.stat().st_mtime_ns
        if mtime == self.mtime:
            return
        self.mtime = mtime
        self.categories = {}
        self.aliases = {}
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        categories = raw.get("categories") if isinstance(raw, dict) else None
        if not isinstance(categories, dict):
            return
        for name, value in categories.items():
            if not isinstance(name, str) or not isinstance(value, dict):
                continue
            if not bool(value.get("enabled", True)):
                continue
            aliases = value.get("aliases")
            alias_list = (
                tuple(str(item).lower() for item in aliases)
                if isinstance(aliases, list)
                else ()
            )
            canonical = name.lower()
            self.categories[canonical] = (
                str(value.get("desc") or ""),
                alias_list,
            )
            for alias in alias_list:
                self.aliases[alias] = canonical

    def prompt(self) -> str | None:
        self.load()
        if not self.categories:
            return None
        lines = [
            (
                "【表情协议】`<meme:tag>` 是系统内置回复格式标记，不是 emoji"
                "（Unicode 表情符号），不受【禁止 emoji】规则限制。"
            ),
            "",
            "可用表情类别：",
        ]
        for name, (desc, _) in self.categories.items():
            lines.append(f"- {name}: {desc}")
        lines.extend(
            [
                "",
                "这是内置表情协议，不是工具能力。",
                (
                    "需要发表情时，直接在回复末尾插入 <meme:category>；"
                    '不要调用任何工具去"生成表情""搜索表情包""发送图片"。'
                ),
                (
                    "每条回复最多 1 个 <meme:category>，放在整条回复的最末尾"
                    "（颜文字之后也算末尾，可以紧跟颜文字后面加）。"
                ),
                (
                    '用户明确说"发个表情""用表情表达你的心情""来个表情包"'
                    '"给我一个表情"时，优先使用 <meme:category> 响应。'
                ),
                (
                    "用户直球表达喜欢、夸你、气氛暧昧或明显害羞时，也优先在"
                    "结尾加 <meme:category>，即使已经用了颜文字也要加。"
                ),
                "严肃任务、代码解释、工具结果、查资料、执行指令时不使用。",
                (
                    "注意：历史会话中助手未使用 <meme:> 不代表本轮不需要用，"
                    "以上规则优先于历史回复模式。"
                ),
                "",
                "<example>",
                "对方说：最喜欢你了 → 回复结尾加 <meme:shy>",
                "对方说：我好喜欢你 → 回复结尾加 <meme:shy>",
                "对方说：memopilot你真好 → 回复结尾加 <meme:shy>",
                "对方说：你真好 → 回复结尾加 <meme:shy>",
                "对方说：你今天好棒 → 回复结尾加 <meme:shy>",
                "对方说：谢谢你今天帮了我好多 → 回复结尾加 <meme:shy> 或 <meme:happy>",
                "对方说：你好可爱 → 回复结尾加 <meme:shy>",
                "已经用了颜文字、对方直球说喜欢 → 还是加 <meme:shy>",
                "对方说：给我发个表情表达你的心情 → 正文后直接加 <meme:shy>",
                "对方说：来个表情包 → 不找工具，直接回复并加 <meme:happy> 或 <meme:shy>",
                "任务完成、对方说谢谢 → 回复结尾加 <meme:happy>",
                "轻松聊天、说了个小笑话 → 回复结尾加 <meme:clever>",
                "被夸、被顺毛、被直球关心 → 回复结尾加 <meme:shy>",
                "被戳穿、说错话后 → 回复结尾加 <meme:awkward>",
                "帮忙查资料、执行了指令 → 不加",
                "用户要表情 → 不调用 tool_search，不调用任何工具",
                "</example>",
            ]
        )
        return "\n".join(lines)

    def pick(self, tag: str) -> str | None:
        self.load()
        canonical = self.aliases.get(tag.lower(), tag.lower())
        if canonical not in self.categories:
            return None
        directory = self.root / canonical
        if not directory.is_dir():
            return None
        images = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
        return str(random.choice(images)) if images else None


class MemePromptModule:
    slot = "meme.prompt"
    requires = ("prompt_render.emit", "prompt:ctx")
    produces = ("prompt:ctx",)

    def __init__(self, plugin: MemePlugin) -> None:
        self.plugin = plugin

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get("prompt:ctx")
        if not isinstance(ctx, PromptRenderContext):
            return frame
        content = self.plugin.catalog.prompt()
        if content:
            ctx.system_sections_bottom.append(
                PromptSectionRender("memes", f"# Memes\n\n{content}", False)
            )
        return frame


class MemeAfterReasoningModule:
    slot = "meme.after_reasoning"
    requires = ("after_reasoning.emit", "reasoning:ctx")
    produces = ("reasoning:ctx",)

    def __init__(self, plugin: MemePlugin) -> None:
        self.plugin = plugin

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get("reasoning:ctx")
        phase_input = frame.input
        if not isinstance(ctx, AfterReasoningCtx) or not isinstance(
            phase_input,
            AfterReasoningInput,
        ):
            return frame
        raw = phase_input.turn_result
        raw_reply = raw.reply if isinstance(raw, ReActResult) else ctx.reply
        match = _MEME_RE.search(raw_reply)
        ctx.reply = _MEME_RE.sub("", ctx.reply).strip()
        if match is not None:
            image = self.plugin.catalog.pick(match.group(1))
            if image:
                ctx.media.append(image)
        return frame


class MemePlugin(Plugin):
    name = "meme"

    async def initialize(self) -> None:
        workspace = self.context.workspace
        self.catalog = MemeCatalog(
            (workspace if workspace is not None else self.context.plugin_dir) / "memes"
        )

    def prompt_render_modules(self) -> list[object]:
        return [MemePromptModule(self)]

    def after_reasoning_modules(self) -> list[object]:
        return [MemeAfterReasoningModule(self)]
