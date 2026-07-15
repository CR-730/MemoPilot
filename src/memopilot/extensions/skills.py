"""Markdown Skills 的安全发现、可用性检查与提示注入。"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    skill_path: Path
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SkillLoadResult:
    skills: tuple[SkillDefinition, ...]
    diagnostics: tuple[SkillDiagnostic, ...]


class SkillLoader:
    def __init__(
        self,
        *,
        builtin_root: Path | None = None,
        workspace_root: Path | None = None,
        available_tools: frozenset[str] = frozenset(),
        max_bytes: int = 32 * 1024,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("Skill 文件大小上限必须大于 0")
        self._builtin_root = builtin_root
        self._workspace_root = workspace_root
        self._available_tools = available_tools
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
                    resolved.relative_to(resolved_root)
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
                    loaded[skill.name] = SkillDefinition(
                        name=skill.name,
                        description=skill.description,
                        background_allowed=skill.background_allowed,
                        required_tools=skill.required_tools,
                        content=skill.content,
                        source=source,
                        path=resolved,
                        available=not missing_tools,
                        missing_tools=missing_tools,
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
        names = tuple(match.group(1).lower() for match in _MENTION.finditer(text))
        return self.render(names, max_chars=max_chars)

    def render(self, names: tuple[str, ...], *, max_chars: int) -> str:
        if max_chars <= 0:
            raise ValueError("Skill Prompt 预算必须大于 0")
        blocks = []
        for name in sorted(set(names)):
            skill = self._skills.get(name)
            if skill is None or not skill.available:
                continue
            blocks.append(f"# Skill: {skill.name}\n{skill.content.strip()}")
        return "\n\n".join(blocks)[:max_chars]

    def background_candidates(self) -> tuple[SkillDefinition, ...]:
        return tuple(
            skill
            for skill in self._skills.values()
            if skill.background_allowed and skill.available
        )


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
    required = {"name", "description", "background_allowed", "required_tools"}
    missing = required.difference(raw_metadata)
    unknown = set(raw_metadata).difference(required)
    if missing:
        raise _SkillError("invalid_skill", "缺少字段: " + ", ".join(sorted(missing)))
    if unknown:
        raise _SkillError("invalid_skill", "未知字段: " + ", ".join(sorted(unknown)))
    name = str(raw_metadata["name"])
    if not _SKILL_NAME.fullmatch(name):
        raise _SkillError("invalid_skill", f"Skill name 格式无效: {name}")
    description = str(raw_metadata["description"]).strip()
    if not description:
        raise _SkillError("invalid_skill", "Skill description 不能为空")
    background_allowed = raw_metadata["background_allowed"]
    if not isinstance(background_allowed, bool):
        raise _SkillError("invalid_skill", "background_allowed 必须是布尔值")
    required_tools = _string_tuple(raw_metadata["required_tools"])
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
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _SkillError("invalid_skill", "required_tools 必须是字符串数组")
    return tuple(value)


__all__ = [
    "SkillCatalog",
    "SkillDefinition",
    "SkillDiagnostic",
    "SkillLoadResult",
    "SkillLoader",
]

