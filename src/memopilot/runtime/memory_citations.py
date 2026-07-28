"""飞书 live 流式输出中的内部记忆引用前缀清理。"""

from __future__ import annotations

_CITATION_MARKER = "§cited:["


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


__all__ = ["visible_response_prefix"]
