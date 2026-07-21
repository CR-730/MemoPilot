"""Markdown Skills 的安全发现、可用性检查与提示注入。"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MENTION = re.compile(r"\$([a-z][a-z0-9_-]{0,63})", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    description: str
    background_allowed: bool
    required_tools: tuple[str, ...]
    content: str
    source: str
    path: Path | None
    available: bool = True
    missing_tools: tuple[str, ...] = ()
    always: bool = False
    required_bins: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()
    missing_bins: tuple[str, ...] = ()
    missing_env: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    skill_path: Path
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SkillLoadResult:
    skills: tuple[SkillDefinition, ...]
    diagnostics: tuple[SkillDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ActiveSkillBlock:
    """一个必须整体注入或整体省略的激活 Skill 正文。"""

    name: str
    content: str


@dataclass(frozen=True, slots=True)
class SkillTurnPrompt:
    """将低优先级目录与高优先级激活正文显式分开。"""

    catalog: str
    active: tuple[ActiveSkillBlock, ...]


class SkillLoader:
    def __init__(
        self,
        *,
        builtin_root: Path | None = None,
        workspace_root: Path | None = None,
        available_tools: frozenset[str] = frozenset(),
        binary_checker: Callable[[str], bool] | None = None,
        environment: Mapping[str, str] | None = None,
        max_bytes: int = 32 * 1024,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("Skill 文件大小上限必须大于 0")
        self._builtin_root = builtin_root
        self._workspace_root = workspace_root
        self._available_tools = available_tools
        self._binary_checker = binary_checker or (lambda name: shutil.which(name) is not None)
        self._environment = os.environ if environment is None else environment
        self._max_bytes = max_bytes

    def load(self) -> SkillLoadResult:
        loaded: dict[str, SkillDefinition] = {}
        diagnostics: list[SkillDiagnostic] = []
        for source, root in (
            ("builtin", self._builtin_root),
            ("workspace", self._workspace_root),
        ):
            if root is None or not root.exists():
                continue
            resolved_root = root.resolve()
            for directory in sorted(path for path in root.iterdir() if path.is_dir()):
                path = directory / "SKILL.md"
                if not path.exists():
                    continue
                try:
                    resolved = path.resolve(strict=True)
                    try:
                        resolved.relative_to(resolved_root)
                    except ValueError as exc:
                        raise _SkillError(
                            "path_outside_root", "Skill 路径不能越过发现根目录"
                        ) from exc
                    raw = resolved.read_bytes()
                    if len(raw) > self._max_bytes:
                        raise _SkillError("skill_too_large", "Skill 文件超过大小上限")
                    text = raw.decode("utf-8")
                    skill = _parse_skill(text, source=source, path=resolved)
                    if skill.name != directory.name:
                        raise _SkillError(
                            "invalid_skill",
                            "frontmatter name 必须与目录名一致",
                        )
                    missing_tools = tuple(
                        sorted(set(skill.required_tools).difference(self._available_tools))
                    )
                    missing_bins = tuple(
                        name for name in skill.required_bins if not self._binary_checker(name)
                    )
                    missing_env = tuple(
                        name for name in skill.required_env if not self._environment.get(name)
                    )
                    loaded[skill.name] = SkillDefinition(
                        name=skill.name,
                        description=skill.description,
                        background_allowed=skill.background_allowed,
                        required_tools=skill.required_tools,
                        content=skill.content,
                        source=source,
                        path=resolved,
                        available=not (missing_tools or missing_bins or missing_env),
                        missing_tools=missing_tools,
                        always=skill.always,
                        required_bins=skill.required_bins,
                        required_env=skill.required_env,
                        missing_bins=missing_bins,
                        missing_env=missing_env,
                    )
                except _SkillError as exc:
                    diagnostics.append(SkillDiagnostic(path, exc.code, str(exc)))
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                    diagnostics.append(
                        SkillDiagnostic(path, "invalid_skill", f"{type(exc).__name__}: {exc}")
                    )
        return SkillLoadResult(
            tuple(loaded[name] for name in sorted(loaded)),
            tuple(diagnostics),
        )


class SkillCatalog:
    def __init__(self, skills: tuple[SkillDefinition, ...] = ()) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"Skill 重复: {skill.name}")
            self._skills[skill.name] = skill

    def render_mentions(self, text: str, *, max_chars: int) -> str:
        names = self._mentioned_names(text)
        return self.render(names, max_chars=max_chars)

    def build_turn_prompt(self, text: str) -> SkillTurnPrompt:
        active_names = {
            skill.name
            for skill in self._skills.values()
            if skill.always and skill.available
        }
        active_names.update(self._mentioned_names(text))
        blocks = tuple(
            ActiveSkillBlock(name, self._render_block(skill))
            for name in sorted(active_names)
            if (skill := self._skills.get(name)) is not None and skill.available
        )
        return SkillTurnPrompt(catalog=self.render_catalog(), active=blocks)

    def render_turn(self, text: str, *, max_chars: int) -> str:
        """兼容调用：先原子纳入激活正文，再用剩余预算裁剪目录。"""
        if max_chars <= 0:
            raise ValueError("Skill Prompt 预算必须大于 0")
        prompt = self.build_turn_prompt(text)
        active = _render_atomic_blocks(prompt.active, max_chars=max_chars)
        return _append_truncated(active, prompt.catalog, max_chars=max_chars)

    def render_catalog(self) -> str:
        if not self._skills:
            return ""
        lines = ["# Skills Catalog"]
        for name in sorted(self._skills):
            skill = self._skills[name]
            availability = "available"
            missing = (
                *(f"TOOL: {item}" for item in skill.missing_tools),
                *(f"CLI: {item}" for item in skill.missing_bins),
                *(f"ENV: {item}" for item in skill.missing_env),
            )
            if missing:
                availability = "unavailable: " + ", ".join(missing)
            lines.append(
                f"- {skill.name} | {skill.description} | {skill.source} | {availability}"
            )
        return "\n".join(lines)

    def render(self, names: tuple[str, ...], *, max_chars: int) -> str:
        if max_chars <= 0:
            raise ValueError("Skill Prompt 预算必须大于 0")
        blocks: list[ActiveSkillBlock] = []
        for name in sorted(set(names)):
            skill = self._skills.get(name)
            if skill is None or not skill.available:
                continue
            blocks.append(ActiveSkillBlock(name, self._render_block(skill)))
        return _render_atomic_blocks(tuple(blocks), max_chars=max_chars)

    def background_candidates(self) -> tuple[SkillDefinition, ...]:
        return tuple(
            skill
            for skill in self._skills.values()
            if skill.background_allowed and skill.available
        )

    def refresh_available_tools(self, available_tools: frozenset[str]) -> None:
        for name, skill in tuple(self._skills.items()):
            missing = tuple(sorted(set(skill.required_tools).difference(available_tools)))
            self._skills[name] = replace(
                skill,
                available=not (missing or skill.missing_bins or skill.missing_env),
                missing_tools=missing,
            )

    def _mentioned_names(self, text: str) -> tuple[str, ...]:
        return tuple(sorted({match.group(1).lower() for match in _MENTION.finditer(text)}))

    @staticmethod
    def _render_block(skill: SkillDefinition) -> str:
        return f"# Skill: {skill.name}\n{skill.content.strip()}"


def _render_atomic_blocks(
    blocks: tuple[ActiveSkillBlock, ...],
    *,
    max_chars: int,
) -> str:
    rendered = ""
    for block in blocks:
        separator = "\n\n" if rendered else ""
        candidate = rendered + separator + block.content
        if len(candidate) > max_chars:
            continue
        rendered = candidate
    return rendered


def _append_truncated(current: str, content: str, *, max_chars: int) -> str:
    text = content.strip()
    if not text or len(current) >= max_chars:
        return current
    separator = "\n\n" if current else ""
    remaining = max_chars - len(current) - len(separator)
    if remaining <= 0:
        return current
    return current + separator + text[:remaining]


class _SkillError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _parse_skill(text: str, *, source: str, path: Path) -> SkillDefinition:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise _SkillError("invalid_skill", "Skill 缺少 YAML frontmatter")
    end = normalized.find("\n---\n", 4)
    if end < 0:
        raise _SkillError("invalid_skill", "Skill frontmatter 未闭合")
    raw_metadata = yaml.safe_load(normalized[4:end])
    if not isinstance(raw_metadata, dict):
        raise _SkillError("invalid_skill", "Skill frontmatter 必须是对象")
    required = {"name", "description"}
    missing = required.difference(raw_metadata)
    if missing:
        raise _SkillError("invalid_skill", "缺少字段: " + ", ".join(sorted(missing)))
    name = str(raw_metadata["name"])
    if not _SKILL_NAME.fullmatch(name):
        raise _SkillError("invalid_skill", f"Skill name 格式无效: {name}")
    description = str(raw_metadata["description"]).strip()
    if not description:
        raise _SkillError("invalid_skill", "Skill description 不能为空")
    background_allowed = raw_metadata.get("background_allowed", False)
    if not isinstance(background_allowed, bool):
        raise _SkillError("invalid_skill", "background_allowed 必须是布尔值")
    required_tools = _string_tuple(raw_metadata.get("required_tools", []), "required_tools")
    config = _skill_config(raw_metadata.get("metadata"))
    always = raw_metadata.get("always", config.get("always", False))
    if not isinstance(always, bool):
        raise _SkillError("invalid_skill", "always 必须是布尔值")
    requires = config.get("requires", {})
    if not isinstance(requires, dict):
        raise _SkillError("invalid_skill", "requires 必须是对象")
    required_bins = _string_tuple(requires.get("bins", []), "requires.bins")
    required_env = _string_tuple(requires.get("env", []), "requires.env")
    content = normalized[end + 5 :].strip()
    if not content:
        raise _SkillError("invalid_skill", "Skill 正文不能为空")
    return SkillDefinition(
        name,
        description,
        background_allowed,
        required_tools,
        content,
        source,
        path,
        always=always,
        required_bins=required_bins,
        required_env=required_env,
    )


def _skill_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    for key in ("memopilot", "skill"):
        nested = value.get(key)
        if isinstance(nested, dict):
            return nested
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _SkillError("invalid_skill", f"{field} 必须是字符串数组")
    return tuple(value)


__all__ = [
    "ActiveSkillBlock",
    "SkillCatalog",
    "SkillDefinition",
    "SkillDiagnostic",
    "SkillLoadResult",
    "SkillLoader",
    "SkillTurnPrompt",
]
