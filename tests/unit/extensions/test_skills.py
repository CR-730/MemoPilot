from __future__ import annotations

from pathlib import Path

from memopilot.extensions.skills import SkillCatalog, SkillLoader


def _write_skill(
    root: Path,
    directory: str,
    *,
    name: str,
    description: str = "测试 Skill",
    background_allowed: bool = False,
    required_tools: tuple[str, ...] = (),
    content: str = "按步骤执行。",
) -> None:
    path = root / directory
    path.mkdir(parents=True)
    tools = "\n".join(f"  - {tool}" for tool in required_tools) or "  []"
    (path / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
background_allowed: {str(background_allowed).lower()}
required_tools:
{tools}
---
{content}
""",
        encoding="utf-8",
    )


def test_workspace_skill_overrides_builtin_and_mentions_render_full_content(
    tmp_path: Path,
) -> None:
    builtin = tmp_path / "builtin"
    workspace = tmp_path / "workspace"
    _write_skill(builtin, "review", name="review", content="旧流程")
    _write_skill(workspace, "review", name="review", content="新流程")

    loaded = SkillLoader(
        builtin_root=builtin,
        workspace_root=workspace,
        available_tools=frozenset(),
    ).load()
    catalog = SkillCatalog(loaded.skills)

    assert [skill.name for skill in loaded.skills] == ["review"]
    assert loaded.skills[0].source == "workspace"
    assert catalog.render_mentions("请用 $review 检查", max_chars=200) == (
        "# Skill: review\n新流程"
    )


def test_background_candidates_require_permission_and_available_tools(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(
        root,
        "allowed",
        name="allowed",
        background_allowed=True,
        required_tools=("search",),
    )
    _write_skill(root, "manual", name="manual", background_allowed=False)
    _write_skill(
        root,
        "missing",
        name="missing",
        background_allowed=True,
        required_tools=("email",),
    )

    loaded = SkillLoader(
        workspace_root=root,
        available_tools=frozenset({"search"}),
    ).load()
    catalog = SkillCatalog(loaded.skills)

    assert [skill.name for skill in catalog.background_candidates()] == ["allowed"]
    missing = next(skill for skill in loaded.skills if skill.name == "missing")
    assert missing.available is False
    assert missing.missing_tools == ("email",)


def test_invalid_and_oversized_skills_are_diagnosed_without_blocking_others(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "valid", name="valid")
    invalid = root / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text("没有 frontmatter", encoding="utf-8")
    _write_skill(root, "large", name="large", content="x" * 200)

    loaded = SkillLoader(workspace_root=root, max_bytes=128).load()

    assert [skill.name for skill in loaded.skills] == ["valid"]
    assert {(item.skill_path.name, item.code) for item in loaded.diagnostics} == {
        ("SKILL.md", "invalid_skill"),
        ("SKILL.md", "skill_too_large"),
    }


def test_skill_prompt_budget_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "a", name="a", content="AAAA")
    _write_skill(root, "b", name="b", content="BBBB")
    catalog = SkillCatalog(SkillLoader(workspace_root=root).load().skills)

    assert catalog.render(("b", "a"), max_chars=24) == "# Skill: a\nAAAA\n\n# Skill"
