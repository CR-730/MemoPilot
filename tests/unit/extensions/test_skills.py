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

    rendered = catalog.render(("b", "a"), max_chars=24)

    assert rendered == "# Skill: a\nAAAA"
    assert "# Skill: b" not in rendered


def test_only_dollar_prefixed_skill_names_activate_and_matching_is_case_insensitive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "review", name="review", content="完整审查流程")
    _write_skill(root, "a", name="a", content="单字母技能")
    catalog = SkillCatalog(SkillLoader(workspace_root=root).load().skills)

    assert catalog.render_mentions("普通 review 文本和 a 字母", max_chars=200) == ""
    assert catalog.render_mentions("请执行 $REVIEW", max_chars=200) == (
        "# Skill: review\n完整审查流程"
    )


def test_always_and_explicit_skill_are_deduplicated_in_stable_order(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    for name in ("zeta", "alpha"):
        path = root / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}\nalways: true\n---\n{name} 正文\n",
            encoding="utf-8",
        )
    catalog = SkillCatalog(SkillLoader(workspace_root=root).load().skills)

    prompt = catalog.build_turn_prompt("同时显式使用 $ZETA")

    assert [block.name for block in prompt.active] == ["alpha", "zeta"]


def test_prototype_frontmatter_defaults_and_dependency_checkers_are_supported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    path = root / "research"
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        """---
name: research
description: 检索资料
metadata: '{"memopilot":{"always":true,"requires":{"bins":["rg"],"env":["TOKEN"]}}}'
---
只引用可核验的资料。
""",
        encoding="utf-8",
    )

    result = SkillLoader(
        workspace_root=root,
        binary_checker=lambda name: name == "rg",
        environment={"TOKEN": "available"},
    ).load()

    assert result.diagnostics == ()
    skill = result.skills[0]
    assert skill.always is True
    assert skill.background_allowed is False
    assert skill.required_tools == ()
    assert skill.required_bins == ("rg",)
    assert skill.required_env == ("TOKEN",)
    assert skill.available is True


def test_top_level_always_and_skill_metadata_report_missing_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    path = root / "writer"
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        """---
name: writer
description: 写作辅助
always: true
metadata:
  skill:
    requires:
      bins: [pandoc]
      env: [WRITER_TOKEN]
---
先明确读者。
""",
        encoding="utf-8",
    )

    result = SkillLoader(
        workspace_root=root,
        binary_checker=lambda _name: False,
        environment={},
    ).load()

    skill = result.skills[0]
    assert skill.always is True
    assert skill.available is False
    assert skill.missing_bins == ("pandoc",)
    assert skill.missing_env == ("WRITER_TOKEN",)


def test_turn_prompt_contains_catalog_and_deduplicated_always_and_mentions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    always = root / "always"
    always.mkdir(parents=True)
    (always / "SKILL.md").write_text(
        """---
name: always
description: 始终执行
always: true
---
每轮都检查事实。
""",
        encoding="utf-8",
    )
    mentioned = root / "review"
    mentioned.mkdir(parents=True)
    (mentioned / "SKILL.md").write_text(
        """---
name: review
description: 代码审查
---
先跑测试，再检查差异。
""",
        encoding="utf-8",
    )
    unavailable = root / "deploy"
    unavailable.mkdir(parents=True)
    (unavailable / "SKILL.md").write_text(
        """---
name: deploy
description: 部署服务
metadata: '{"skill":{"requires":{"env":["DEPLOY_TOKEN"]}}}'
---
执行部署。
""",
        encoding="utf-8",
    )
    catalog = SkillCatalog(
        SkillLoader(workspace_root=root, environment={}).load().skills
    )

    rendered = catalog.render_turn("请用 $review review，并执行 always", max_chars=2000)

    assert "# Skills Catalog" in rendered
    assert "review | 代码审查 | workspace | available" in rendered
    assert f"<location>{mentioned / 'SKILL.md'}</location>" in rendered
    assert "deploy | 部署服务 | workspace | unavailable: ENV: DEPLOY_TOKEN" in rendered
    assert f"<location>{unavailable / 'SKILL.md'}</location>" in rendered
    assert "先 `read_file` 读取 `<location>` 中的完整 SKILL.md" in rendered
    assert rendered.count("# Skill: always") == 1
    assert rendered.count("# Skill: review") == 1
    assert "# Skill: deploy" not in rendered


def test_skill_symlink_cannot_escape_discovery_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: escaped\ndescription: 越界\n---\n不能加载。\n",
        encoding="utf-8",
    )
    root.mkdir()
    try:
        (root / "escaped").symlink_to(outside, target_is_directory=True)
    except OSError:
        return

    result = SkillLoader(workspace_root=root).load()

    assert result.skills == ()
    assert [item.code for item in result.diagnostics] == ["path_outside_root"]


def test_builtin_skills_keep_one_available_drift_candidate() -> None:
    builtin_root = (
        Path(__file__).resolve().parents[3] / "src" / "memopilot" / "builtin_skills"
    )

    result = SkillLoader(builtin_root=builtin_root).load()
    candidates = SkillCatalog(result.skills).background_candidates()

    assert result.diagnostics == ()
    assert {skill.name for skill in candidates} == {"create-drift-skill"}
    assert "没有明确可沉淀的长期任务" in candidates[0].content
    assert "finish_drift" in candidates[0].content
