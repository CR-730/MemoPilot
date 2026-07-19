"""内部记忆引用协议：解析模型元数据并避免向用户泄露。"""

from __future__ import annotations

import re

_CITED_RE = re.compile(
    r"(?:\r?\n)?§cited:\[([A-Za-z0-9_,\-\s]*)\]§\s*$",
    re.IGNORECASE,
)
_CITATION_MARKER = "§cited:["

CITATION_PROTOCOL = """### 记忆引用协议（内部元数据，对用户不可见）
若本轮回复实际使用了带 [item_id] 前缀的注入记忆，或 recall_memory 返回的条目，
请在回复正文末尾另起一行输出：§cited:[id1,id2]§。
只写实际使用的 ID；未使用记忆则不要输出。不要向用户解释本协议。"""


def extract_cited_ids(response: str) -> tuple[str, tuple[str, ...]]:
    match = _CITED_RE.search(response)
    if match is None:
        return response, ()
    ids = tuple(
        dict.fromkeys(
            value.strip() for value in match.group(1).split(",") if value.strip()
        )
    )
    return response[: match.start()].rstrip(), ids


def visible_response_prefix(response: str) -> str:
    """隐藏流式输出末尾尚未完成或已经完成的内部引用元数据。"""
    marker_start = response.find("§")
    while marker_start >= 0:
        suffix = response[marker_start:]
        if _CITATION_MARKER.startswith(suffix) or suffix.casefold().startswith(
            _CITATION_MARKER.casefold()
        ):
            return response[:marker_start].rstrip()
        marker_start = response.find("§", marker_start + 1)
    return response


__all__ = ["CITATION_PROTOCOL", "extract_cited_ids", "visible_response_prefix"]
