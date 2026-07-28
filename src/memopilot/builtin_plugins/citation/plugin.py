from __future__ import annotations

import json
import re
from typing import Any, cast

from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import AfterReasoningCtx
from memopilot.extensions.prompts import PromptRenderContext, PromptSectionRender

_PROMPT_CTX_SLOT = "prompt:ctx"
_REASONING_CTX_SLOT = "reasoning:ctx"
_PERSIST_CITED_SLOT = "persist:assistant:cited_memory_ids"
_TRAILING_PROTOCOL_TAG = r"<[a-zA-Z][a-zA-Z0-9_-]*:[^<>\s]+>"
_CITED_RE = re.compile(
    rf"(?:\r?\n)?§cited:\[([A-Za-z0-9_,\-\s]*)\]§"
    rf"(?P<trailing>(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)*)$",
    re.IGNORECASE,
)
_TRAILING_PROTOCOL_TAGS_RE = re.compile(
    rf"(?:\s*{_TRAILING_PROTOCOL_TAG}\s*)+$",
    re.IGNORECASE,
)
_INLINE_MEMORY_REF_RE = re.compile(
    r"[ \t]*(?:\[§[A-Za-z0-9:_-]{1,128}\])+",
    re.IGNORECASE,
)

_CITATION_PROTOCOL = (
    "### 记忆引用协议 - 内部元数据，对用户不可见\n"
    "每轮回复若用到了系统注入的记忆条目 [item_id] 前缀标识，或 "
    "recall_memory / fetch_messages 工具返回的条目，在回复正文末尾另起一行输出：\n"
    "§cited:[id1,id2,id3]§\n"
    "格式规则：§ 包裹，英文逗号分隔，无空格，只写 ID，不含其他内容。\n"
    "若本轮未引用任何记忆条目，不输出此行。\n"
    "绝对不要在正文里提及这行的存在，不要向用户解释引用了什么，不要说根据记忆。\n"
    "你了解用户的事是因为你们相处了很久，直接说你上次、我记得，不要暴露内部机制。"
)


class CitationPromptModule:
    slot = "citation.prompt"
    requires = ("prompt_render.emit", _PROMPT_CTX_SLOT)
    produces = (_PROMPT_CTX_SLOT,)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_PROMPT_CTX_SLOT)
        if not isinstance(ctx, PromptRenderContext):
            return frame
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                name="citation_protocol",
                content=_CITATION_PROTOCOL,
                is_static=True,
            )
        )
        return frame


class CitationAfterReasoningModule:
    slot = "citation.after_reasoning"
    requires = ("after_reasoning.build_ctx", _REASONING_CTX_SLOT)
    produces = (_REASONING_CTX_SLOT, _PERSIST_CITED_SLOT)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_REASONING_CTX_SLOT)
        if not isinstance(ctx, AfterReasoningCtx):
            return frame
        reply = ctx.reply
        citation_cleaned, cited_ids = extract_cited_ids(reply)
        has_explicit_marker = citation_cleaned != reply
        cleaned = strip_inline_memory_refs(citation_cleaned)
        if has_explicit_marker:
            candidates = cited_ids
        else:
            candidates = extract_cited_ids_from_tool_chain(list(ctx.tool_chain))
        frame.slots[_PERSIST_CITED_SLOT] = candidates
        if cleaned != reply:
            ctx.reply = cleaned
        return frame


class ProtocolTagCleanupModule:
    slot = "citation.protocol_cleanup"
    requires = ("after_reasoning.emit", _REASONING_CTX_SLOT)
    produces = (_REASONING_CTX_SLOT,)

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_REASONING_CTX_SLOT)
        if not isinstance(ctx, AfterReasoningCtx):
            return frame
        cleaned = strip_inline_memory_refs(strip_trailing_protocol_tags(ctx.reply))
        if cleaned != ctx.reply:
            ctx.reply = cleaned
        return frame


class CitationPlugin(Plugin):
    name = "citation"
    version = "0.1.0"
    desc = "注入并消费内部记忆引用协议"

    def prompt_render_modules(self) -> list[object]:
        return [CitationPromptModule()]

    def after_reasoning_modules(self) -> list[object]:
        return [CitationAfterReasoningModule(), ProtocolTagCleanupModule()]


def extract_cited_ids(response: str) -> tuple[str, list[str]]:
    match = _CITED_RE.search(response)
    if not match:
        return response, []
    ids = [item.strip() for item in match.group(1).split(",") if item.strip()]
    trailing = match.group("trailing").strip()
    clean = response[: match.start()].rstrip()
    if trailing:
        clean = f"{clean} {trailing}".strip()
    return clean, ids


def strip_trailing_protocol_tags(response: str) -> str:
    return _TRAILING_PROTOCOL_TAGS_RE.sub("", response).rstrip()


def strip_inline_memory_refs(response: str) -> str:
    return _INLINE_MEMORY_REF_RE.sub("", response).rstrip()


def extract_cited_ids_from_tool_chain(
    tool_chain: list[dict[str, object]],
) -> list[str]:
    cited: list[str] = []
    seen: set[str] = set()
    for snapshot in tool_chain:
        if (
            snapshot.get("name") != "recall_memory"
            or snapshot.get("status") != "success"
        ):
            continue
        raw_result = snapshot.get("result")
        if isinstance(raw_result, dict):
            data = cast(dict[str, object], raw_result)
        elif isinstance(raw_result, str):
            try:
                decoded = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(decoded, dict):
                continue
            data = cast(dict[str, object], decoded)
        else:
            continue
        raw_ids: list[object] = []
        cited_ids = data.get("cited_item_ids")
        if isinstance(cited_ids, list):
            raw_ids.extend(cast(list[object], cited_ids))
        else:
            items = data.get("items")
            if isinstance(items, list):
                for raw_item in cast(list[object], items):
                    if isinstance(raw_item, dict):
                        raw_ids.append(raw_item.get("id"))
        for raw_id in raw_ids:
            item_id = str(raw_id or "").strip()
            if item_id and item_id not in seen:
                seen.add(item_id)
                cited.append(item_id)
    return cited
