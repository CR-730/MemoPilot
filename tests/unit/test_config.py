from __future__ import annotations

from pathlib import Path

import pytest

from memopilot.config import MemoPilotSettings, load_settings


def test_settings_use_documented_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    settings = MemoPilotSettings(_env_file=None)

    assert settings.chat_base_url == "https://api.deepseek.com"
    assert settings.chat_model == "deepseek-v4-flash"
    assert settings.wake_tick_seconds == 300
    assert settings.content_half_life_hours == 6
    assert settings.proactive_cooldown_hours == 2
    assert settings.lease_ttl_seconds == 30
    assert settings.lease_heartbeat_seconds == 10
    assert settings.reclaim_idle_seconds == 60
    assert settings.sqlite_busy_timeout_seconds == 5
    assert settings.mcp_startup_timeout_seconds == 15
    assert settings.mcp_call_timeout_seconds == 30
    assert settings.llm_retry_limit == 2
    assert settings.llm_max_iterations == 10
    assert settings.llm_max_output_tokens == 2048
    assert settings.llm_timeout_seconds == 60
    assert settings.llm_thinking_enabled is False


def test_environment_overrides_dotenv_and_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "wake_tick_seconds: 900\nredis_url: redis://yaml:6379/0\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MEMOPILOT_WAKE_TICK_SECONDS=600\nMEMOPILOT_REDIS_URL=redis://dotenv:6379/0\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMOPILOT_WAKE_TICK_SECONDS", "300")

    settings = MemoPilotSettings(_env_file=env_file)

    assert settings.wake_tick_seconds == 300
    assert settings.redis_url == "redis://dotenv:6379/0"


def test_feishu_allowlist_accepts_comma_separated_environment_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMOPILOT_FEISHU_ALLOW_FROM", "ou_owner, user_owner ,ou_owner")

    settings = MemoPilotSettings(_env_file=None)

    assert settings.feishu_allow_from == ("ou_owner", "user_owner")


def test_workspace_paths_are_resolved_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    settings = MemoPilotSettings(workspace=workspace, _env_file=None)

    assert settings.workspace == workspace.resolve()
    assert settings.operational_database == workspace.resolve() / "data" / "operational.db"
    assert settings.memory_database == workspace.resolve() / "data" / "memory2.db"
    assert settings.wake_database == workspace.resolve() / "data" / "wake.db"


def test_path_outside_workspace_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="必须位于 workspace 内"):
        MemoPilotSettings(
            workspace=tmp_path / "workspace",
            data_dir=tmp_path / "outside",
            _env_file=None,
        )


def test_runtime_validation_reports_all_missing_required_configuration(tmp_path: Path) -> None:
    settings = MemoPilotSettings(workspace=tmp_path, _env_file=None)

    with pytest.raises(ValueError) as exc_info:
        settings.validate_runtime_ready()

    message = str(exc_info.value)
    assert "MEMOPILOT_CHAT_API_KEY" in message
    assert "MEMOPILOT_EMBEDDING_BASE_URL" in message
    assert "MEMOPILOT_EMBEDDING_MODEL" in message
    assert "MEMOPILOT_EMBEDDING_API_KEY" in message
    assert "MEMOPILOT_EMBEDDING_DIMENSION" in message
    assert "MEMOPILOT_FEISHU_APP_ID" in message
    assert "MEMOPILOT_FEISHU_APP_SECRET" in message
    assert "MEMOPILOT_FEISHU_ALLOW_FROM" not in message


def test_runtime_validation_accepts_complete_configuration(tmp_path: Path) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path,
        chat_api_key="chat-secret",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_api_key="embedding-secret",
        embedding_dimension=1024,
        feishu_app_id="cli_app",
        feishu_app_secret="feishu-secret",
        feishu_allow_from=("ou_owner",),
        _env_file=None,
    )

    settings.validate_runtime_ready()


def test_prototype_toml_maps_llm_and_feishu_without_copying_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "chat-secret")
    monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret")
    config = tmp_path / "config.toml"
    config.write_text(
        """
[llm]
provider = "deepseek"
[llm.main]
model = "deepseek-chat"
api_key = "${DEEPSEEK_API_KEY}"
base_url = "https://api.deepseek.com/v1"
enable_thinking = true
[agent]
max_tokens = 4096
max_iterations = 12
[channels.feishu]
enabled = true
app_id = "cli-app"
app_secret = "${FEISHU_APP_SECRET}"
receive_mode = "ws"
allow_from = []
channel_name = "feishu_work"
""",
        encoding="utf-8",
    )

    settings = load_settings(config, workspace=tmp_path / "workspace")

    assert settings.chat_model == "deepseek-chat"
    assert settings.chat_api_key.get_secret_value() == "chat-secret"
    assert settings.llm_thinking_enabled is True
    assert settings.llm_max_output_tokens == 4096
    assert settings.llm_max_iterations == 12
    assert settings.feishu_app_id == "cli-app"
    assert settings.feishu_app_secret.get_secret_value() == "feishu-secret"
    assert settings.feishu_allow_from == ()
    assert settings.feishu_channel_name == "feishu_work"


def test_phase3_process_validation_is_split_and_empty_allowlist_is_allowed(
    tmp_path: Path,
) -> None:
    app = MemoPilotSettings(
        workspace=tmp_path / "app",
        feishu_app_id="cli-app",
        feishu_app_secret="secret",
        feishu_allow_from=(),
        _env_file=None,
    )
    worker = MemoPilotSettings(
        workspace=tmp_path / "worker",
        chat_api_key="chat-secret",
        feishu_app_id="cli-app",
        feishu_app_secret="secret",
        _env_file=None,
    )

    app.validate_app_ready()
    worker.validate_worker_ready()
