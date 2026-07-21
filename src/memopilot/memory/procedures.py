"""Procedure 记忆的查询、规则归一化与工具触发匹配。"""

from __future__ import annotations

import re
from collections.abc import Sequence

_ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*")
_NEGATIVE = (
    "不能直接使用",
    "不能直接用",
    "不要直接使用",
    "不要直接用",
    "别直接使用",
    "别直接用",
    "不能先使用",
    "不能先用",
    "不要先使用",
    "不要先用",
    "别先使用",
    "别先用",
    "禁止使用",
    "禁止用",
    "不能使用",
    "不能用",
    "不要使用",
    "不要用",
    "别使用",
    "别用",
)
_POSITIVE = (
    "必须先使用",
    "必须先用",
    "必须使用",
    "必须用",
    "优先使用",
    "优先用",
    "先使用",
    "先用",
    "应先使用",
    "应先用",
    "应该使用",
    "应该用",
    "直接使用",
    "直接用",
)


def build_procedure_rule_schema(
    summary: str,
    tool_requirement: str | None = None,
    steps: Sequence[str] = (),
    rule_schema: dict[str, object] | None = None,
) -> dict[str, list[str]]:
    payload = rule_schema or {}
    required = set(_text_list(payload.get("required_tools")))
    forbidden = set(_text_list(payload.get("forbidden_tools")))
    mentioned = set(_text_list(payload.get("mentioned_tools")))
    texts = [summary, *(str(step) for step in steps)]
    for text in texts:
        mentioned.update(_extract_aliases(text))
        for clause in re.split(r"[，。！？；;\n]", text):
            for alias, prefix in _iter_alias_prefixes(clause):
                if any(prefix.endswith(cue) for cue in _NEGATIVE):
                    forbidden.add(alias)
                elif any(prefix.endswith(cue) for cue in _POSITIVE):
                    required.add(alias)
    requirement = str(tool_requirement or "").strip().casefold()
    if requirement:
        required.add(requirement)
        mentioned.add(requirement)
    forbidden.difference_update(required)
    return {
        "required_tools": sorted(required),
        "forbidden_tools": sorted(forbidden),
        "mentioned_tools": sorted(mentioned),
    }


def _extract_aliases(text: str) -> set[str]:
    matches = list(_ALIAS.finditer(text or ""))
    aliases = {match.group(0).casefold() for match in matches}
    for left, right in zip(matches, matches[1:], strict=False):
        if text[left.end() : right.start()].strip() == "":
            aliases.add(f"{left.group(0).casefold()}_{right.group(0).casefold()}")
    return aliases


def _iter_alias_prefixes(clause: str) -> list[tuple[str, str]]:
    matches = list(_ALIAS.finditer(clause or ""))
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(matches):
        match = matches[index]
        prefix = re.sub(r"\s+", "", clause[max(0, match.start() - 12) : match.start()])
        if index < len(matches) - 1:
            following = matches[index + 1]
            if clause[match.end() : following.start()].strip() == "":
                pairs.append(
                    (
                        f"{match.group(0).casefold()}_{following.group(0).casefold()}",
                        prefix,
                    )
                )
                index += 2
                continue
        pairs.append((match.group(0).casefold(), prefix))
        index += 1
    return pairs


def build_trigger_tags(
    summary: str,
    *,
    tool_requirement: str | None,
    rule_schema: dict[str, object] | None,
) -> dict[str, object]:
    schema = build_procedure_rule_schema(
        summary,
        tool_requirement=tool_requirement,
        rule_schema=rule_schema,
    )
    tools = list(dict.fromkeys(schema["required_tools"] + schema["mentioned_tools"]))
    keywords = [token.casefold() for token in _ALIAS.findall(summary) if len(token) >= 3]
    return {
        "tools": tools,
        "skills": [],
        "keywords": list(dict.fromkeys(keywords)),
        "scope": "tool_triggered" if tools or keywords else "global",
    }


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            str(item).strip().casefold()
            for item in value
            if isinstance(item, str) and str(item).strip()
        }
    )


__all__ = [
    "build_procedure_rule_schema",
    "build_trigger_tags",
]
