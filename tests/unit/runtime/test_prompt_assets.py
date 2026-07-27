from pathlib import Path

from memopilot.runtime.prompt_assets import (
    build_agent_environment_prompt,
    build_passive_system_prompt,
)


def test_passive_prompt_keeps_prototype_language_and_tool_rules(tmp_path: Path) -> None:
    prompt = build_passive_system_prompt(
        workspace=tmp_path,
        session_key="feishu:test-chat",
        received_at=None,
    )

    assert "中文，口语" in prompt
    assert "执行类动作必须走工具" in prompt
    assert "简单问题直接回答" in prompt
    assert "严格只输出用户指定的目标内容" not in prompt
    assert "对用户可见的思考过程与正式回复都使用用户当前使用的语言" not in prompt
    assert "明确要求你长期记住" not in prompt
    assert "必须调用 `memorize`" not in prompt
    assert "成功返回 `item_id`" not in prompt
    assert "# MemoPilot" in prompt


def test_environment_prompt_exposes_real_operating_system_and_shell(
    monkeypatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")

    prompt = build_agent_environment_prompt()

    assert "AMD64" in prompt
