"""工具目录搜索、按需解锁和会话级预加载状态。"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memopilot.runtime.tools import Tool, ToolRegistry

META_TOOL_NAMES: frozenset[str] = frozenset({"tool_search"})


@dataclass(frozen=True)
class ToolMeta:
    risk: str = "read-only"
    always_on: bool = False
    search_hint: str | None = None


@dataclass(frozen=True)
class ToolDocument:
    name: str
    description: str
    risk: str
    always_on: bool
    search_hint: str | None
    source_type: str
    source_name: str

    @classmethod
    def from_tool_and_meta(
        cls,
        tool: Tool,
        meta: ToolMeta,
        *,
        source_type: str,
        source_name: str,
    ) -> ToolDocument:
        return cls(
            name=tool.name,
            description=tool.description,
            risk=meta.risk,
            always_on=meta.always_on,
            search_hint=meta.search_hint,
            source_type=source_type,
            source_name=source_name,
        )


class KeywordSearchBackend:
    """原型的轻量关键词后端，不依赖中文分词库。"""

    def __init__(self) -> None:
        self._documents: dict[str, ToolDocument] = {}

    def rebuild(self, documents: Iterable[ToolDocument]) -> None:
        self._documents = {document.name: document for document in documents}

    def add(self, document: ToolDocument) -> None:
        self._documents[document.name] = document

    def remove(self, name: str) -> None:
        self._documents.pop(name, None)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        excluded_names: AbstractSet[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        risk_filter = set(allowed_risk) if allowed_risk else None
        excluded = excluded_names or set()
        exact = self._documents.get(query)
        if exact is not None and query not in excluded:
            if risk_filter is None or exact.risk in risk_filter:
                return [_result(exact, ["名称:精确匹配"])]
        keywords = _normalize(query)
        if not keywords:
            return []
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for name, document in self._documents.items():
            if name in excluded:
                continue
            if risk_filter is not None and document.risk not in risk_filter:
                continue
            score = _score(document, keywords)
            if score > 0:
                ranked.append((score, name, _result(document, _explain(document, keywords))))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in ranked[:top_k]]


def _normalize(query: str) -> set[str]:
    lowered = query.lower().strip()
    if not lowered:
        return set()
    tokens = {lowered, *lowered.split()}
    tokens.update(segment.strip() for segment in re.split(r"([\u4e00-\u9fff]+)", lowered))
    cjk = [character for character in lowered if "\u4e00" <= character <= "\u9fff"]
    tokens.update(cjk)
    tokens.update(cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1))
    tokens.discard("")
    return tokens


def _score(document: ToolDocument, keywords: set[str]) -> int:
    parts = [part for part in document.name.lower().split("_") if part]
    name = document.name.lower()
    hint = (document.search_hint or "").lower()
    description = document.description.lower()
    is_mcp = document.source_type == "mcp"
    score = 0
    for keyword in keywords:
        if keyword in parts:
            score += 12 if is_mcp else 10
        elif any(keyword in part or part in keyword for part in parts):
            score += 6 if is_mcp else 5
        elif keyword in name:
            score += 3
        if hint and keyword in hint:
            score += 4
        if keyword in description:
            score += 2
    return score


def _explain(document: ToolDocument, keywords: set[str]) -> list[str]:
    parts = [part for part in document.name.lower().split("_") if part]
    name = document.name.lower()
    hint = (document.search_hint or "").lower()
    description = document.description.lower()
    reasons: list[str] = []
    for keyword in sorted(keywords):
        if keyword in parts:
            reasons.append(f"名称精确:{keyword}")
        elif any(keyword in part or part in keyword for part in parts):
            reasons.append(f"名称部分:{keyword}")
        elif keyword in name:
            reasons.append(f"名称:{keyword}")
        if hint and keyword in hint:
            reasons.append(f"提示:{keyword}")
        if keyword in description:
            reasons.append(f"描述:{keyword}")
    return list(dict.fromkeys(reasons))


def _result(document: ToolDocument, why_matched: list[str]) -> dict[str, Any]:
    return {
        "name": document.name,
        "summary": document.description[:120],
        "why_matched": why_matched,
        "risk": document.risk,
        "always_on": document.always_on,
    }


class ToolSearchTool:
    """搜索工具目录；每次执行消费一次当前 Turn 的已可见集合。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._excluded_names: ContextVar[frozenset[str] | None] = ContextVar(
            f"tool_search_excluded_{id(self)}",
            default=None,
        )

    def set_excluded_names(self, names: set[str] | None) -> None:
        self._excluded_names.set(frozenset(names) if names is not None else None)

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
    ) -> str:
        excluded = self._excluded_names.get()
        self._excluded_names.set(None)
        query = (query or "").strip()
        if not query:
            return _json_empty("query 不能为空，请描述需要的功能")
        if query.lower().startswith("select:"):
            return self._select(
                query[7:],
                allowed_risk=allowed_risk,
                excluded=set(excluded) if excluded is not None else None,
            )
        matches = self._registry.search(
            query,
            top_k=min(max(1, int(top_k)), 10),
            allowed_risk=allowed_risk,
            excluded_names=set(excluded) if excluded is not None else None,
        )
        if not matches:
            return _json_empty("没有找到匹配工具，请更换关键词重试")
        unlocked = [str(item["name"]) for item in matches]
        return json.dumps(
            {
                "matched": matches,
                "unlocked": unlocked,
                "already_loaded": [],
                "next_action": "下一轮可直接调用 unlocked 中的工具，无需再次搜索。",
            },
            ensure_ascii=False,
        )

    def _select(
        self,
        raw_names: str,
        *,
        allowed_risk: list[str] | None,
        excluded: set[str] | None,
    ) -> str:
        requested = list(
            dict.fromkeys(
                name.strip() for name in raw_names.split(",") if name.strip()
            )
        )
        if not requested:
            return _json_empty("select: 后需要提供工具名")
        skipped = META_TOOL_NAMES | (excluded or set())
        risk_filter = set(allowed_risk) if allowed_risk else None
        already_loaded: list[str] = []
        unlocked: list[str] = []
        missing: list[str] = []
        blocked: list[str] = []
        for name in requested:
            if name in skipped:
                already_loaded.append(name)
                continue
            document = self._registry.get_document(name)
            if document is None:
                missing.append(name)
            elif risk_filter is not None and document.risk not in risk_filter:
                blocked.append(name)
            else:
                unlocked.append(name)
        result: dict[str, Any] = {
            "matched": self._registry.get_document_results(unlocked),
            "unlocked": unlocked,
            "already_loaded": already_loaded,
        }
        tips: list[str] = []
        if already_loaded:
            tips.append("已加载: " + ", ".join(already_loaded))
        if missing:
            tips.append("未找到: " + ", ".join(missing))
        if blocked:
            tips.append("风险等级不符: " + ", ".join(blocked))
        if unlocked:
            result["next_action"] = "下一轮可直接调用 unlocked 中的工具，无需再次搜索。"
        if tips:
            result["tip"] = "; ".join(tips)
        return json.dumps(result, ensure_ascii=False)


def build_tool_search_tool(registry: ToolRegistry) -> Tool:
    from memopilot.runtime.tools import Tool

    search = ToolSearchTool(registry)
    return Tool(
        name="tool_search",
        description=(
            "搜索并加载当前尚未暴露的工具。已知名称时使用 select:工具名；"
            "不知道名称时使用功能关键词。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "关键词或 select:工具名"},
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
                "allowed_risk": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["read-only", "write", "external-side-effect"],
                    },
                    "description": (
                        "允许的风险等级，不填则不过滤。read-only=只读，"
                        "write=写操作，external-side-effect=外部副作用"
                    ),
                },
            },
            "required": ["query"],
        },
        handler=search.execute,
    )


def format_deferred_tools_hint(deferred: dict[str, object]) -> str:
    lines = ["可按需加载的工具目录（这里只列名称，不代表已加载 Schema）："]
    builtin = deferred.get("builtin")
    if isinstance(builtin, list) and builtin:
        lines.append("- builtin: " + ", ".join(str(name) for name in builtin))
    mcp = deferred.get("mcp")
    if isinstance(mcp, dict):
        for server, names in mcp.items():
            if isinstance(names, list) and names:
                lines.append(
                    f"- mcp[{server}]: " + ", ".join(str(name) for name in names)
                )
    if len(lines) == 1:
        return ""
    lines.append("需要时先调用 tool_search；已知名称可使用 select:工具名。")
    return "\n".join(lines)


def _json_empty(tip: str) -> str:
    return json.dumps(
        {"matched": [], "unlocked": [], "already_loaded": [], "tip": tip},
        ensure_ascii=False,
    )


@dataclass
class ToolDiscoveryState:
    _preloaded: dict[str, OrderedDict[str, None]] = field(default_factory=dict)
    capacity: int = 5

    def get_preloaded(self, session_key: str) -> set[str]:
        return set(self._preloaded.get(session_key, {}))

    def get_preloaded_ordered(self, session_key: str) -> list[str]:
        return list(self._preloaded.get(session_key, {}))

    def unlock_names_from_result(self, result_json: str) -> list[str]:
        try:
            payload = json.loads(result_json)
            if not isinstance(payload, dict):
                return []
        except Exception:
            return []
        raw = payload.get("unlocked")
        if not isinstance(raw, list):
            raw = [
                item.get("name")
                for item in payload.get("matched", [])
                if isinstance(item, dict)
            ]
        return list(dict.fromkeys(item for item in raw if isinstance(item, str) and item))

    def update(self, session_key: str, tools_used: list[str], always_on: set[str]) -> None:
        skipped = always_on | META_TOOL_NAMES
        lru = self._preloaded.setdefault(session_key, OrderedDict())
        for name in tools_used:
            if name in skipped:
                continue
            if name in lru:
                lru.move_to_end(name)
            else:
                lru[name] = None
            while len(lru) > self.capacity:
                lru.popitem(last=False)


__all__ = [
    "KeywordSearchBackend",
    "META_TOOL_NAMES",
    "ToolDiscoveryState",
    "ToolDocument",
    "ToolMeta",
    "ToolSearchTool",
    "build_tool_search_tool",
    "format_deferred_tools_hint",
]
