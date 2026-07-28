from __future__ import annotations

from pathlib import Path

import pytest

from memopilot.config import MemoPilotSettings, load_settings


def test_settings_use_documented_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    settings = MemoPilotSettings(_env_file=None)

    assert settings.workspace == (
        Path.home() / ".memopilot" / "memopilot-workspace"
    ).resolve()
    assert settings.chat_base_url == "https://api.deepseek.com"
    assert settings.chat_model == "deepseek-v4-flash"
    assert settings.proactive_tick_seconds == 1800
    assert settings.proactive_context_probability == 0.3
    assert settings.proactive_active_start_hour == 8
    assert settings.proactive_active_end_hour == 23
    assert settings.proactive_enabled is False
    assert settings.drift_enabled is True
    assert settings.drift_min_interval_hours == 3
    assert settings.lease_ttl_seconds == 30
    assert settings.lease_heartbeat_seconds == 10
    assert settings.reclaim_idle_seconds == 60
    assert settings.sqlite_busy_timeout_seconds == 5
    assert settings.mcp_startup_timeout_seconds == 15
    assert settings.mcp_call_timeout_seconds == 30
    assert settings.llm_retry_limit == 2
    assert settings.llm_max_iterations == 10
    assert settings.llm_max_output_tokens == 2048
    assert settings.llm_context_window_tokens == 1_000_000
    assert settings.llm_timeout_seconds == 60
    assert settings.llm_thinking_enabled is False
    assert settings.tool_search_enabled is True
    assert settings.memory_short_term_message_limit == 12
    assert settings.memory_consolidation_keep_count == 12
    assert settings.memory_consolidation_min_new_messages == 5
    assert settings.memory_score_threshold == 0.45
    assert settings.memory_hotness_alpha == 0.2
    assert settings.memory_hotness_half_life_days == 14
    assert settings.memory_embed_timeout_seconds == 5
    assert settings.memory_score_thresholds == {
        "procedure": 0.58,
        "preference": 0.52,
        "event": 0.45,
        "profile": 0.5,
    }
    assert settings.memory_procedure_guard_enabled is True
    assert settings.memory_inject_max_forced == 3
    assert settings.memory_inject_max_procedure_preference == 4
    assert settings.memory_inject_max_event_profile == 2


def test_environment_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MEMOPILOT_PROACTIVE_TICK_SECONDS=1800\n"
        "MEMOPILOT_DRIFT_MIN_INTERVAL_HOURS=4\n"
        "MEMOPILOT_REDIS_URL=redis://dotenv:6379/0\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMOPILOT_PROACTIVE_TICK_SECONDS", "1800")
    monkeypatch.setenv("MEMOPILOT_DRIFT_MIN_INTERVAL_HOURS", "3")

    settings = MemoPilotSettings(_env_file=env_file)

    assert settings.proactive_tick_seconds == 1800
    assert settings.drift_min_interval_hours == 3
    assert settings.redis_url == "redis://dotenv:6379/0"


def test_localhost_redis_is_normalized_to_ipv4_for_windows_asyncio() -> None:
    settings = MemoPilotSettings(
        redis_url="redis://localhost:6379/0",
        _env_file=None,
    )

    assert settings.redis_url == "redis://127.0.0.1:6379/0"


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
    assert settings.proactive_database == workspace.resolve() / "data" / "proactive.db"


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
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-secret")
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
multimodal = false
[llm.fast]
model = "qwen-turbo"
api_key = "fast-secret"
base_url = "https://fast.example/v1"
[llm.vl]
model = "qwen-vl"
api_key = "vl-secret"
base_url = "https://vl.example/v1"
[agent]
max_tokens = 4096
max_iterations = 12
context_window_tokens = 128000
[agent.tools]
search_enabled = true
[memory.embedding]
model = "text-embedding-v3"
api_key = "${EMBEDDING_API_KEY}"
base_url = "https://embedding.example/v1"
dimension = 1024
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
    assert settings.chat_multimodal is False
    assert settings.fast_model == "qwen-turbo"
    assert settings.fast_api_key.get_secret_value() == "fast-secret"
    assert settings.fast_base_url == "https://fast.example/v1"
    assert settings.vl_model == "qwen-vl"
    assert settings.vl_api_key.get_secret_value() == "vl-secret"
    assert settings.vl_base_url == "https://vl.example/v1"
    assert settings.llm_max_output_tokens == 4096
    assert settings.llm_max_iterations == 12
    assert settings.llm_context_window_tokens == 128_000
    assert settings.tool_search_enabled is True
    assert settings.embedding_model == "text-embedding-v3"
    assert settings.embedding_api_key.get_secret_value() == "embedding-secret"
    assert settings.embedding_base_url == "https://embedding.example/v1"
    assert settings.embedding_dimension == 1024
    assert settings.feishu_app_id == "cli-app"
    assert settings.feishu_app_secret.get_secret_value() == "feishu-secret"
    assert settings.feishu_allow_from == ()
    assert settings.feishu_channel_name == "feishu_work"


def test_app_runtime_validation_allows_empty_allowlist(
    tmp_path: Path,
) -> None:
    app = MemoPilotSettings(
        workspace=tmp_path / "app",
        feishu_app_id="cli-app",
        feishu_app_secret="secret",
        feishu_allow_from=(),
        _env_file=None,
    )
    runtime = MemoPilotSettings(
        workspace=tmp_path / "runtime",
        chat_api_key="chat-secret",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_api_key="embedding-secret",
        embedding_dimension=1024,
        feishu_app_id="cli-app",
        feishu_app_secret="secret",
        _env_file=None,
    )

    app.validate_app_ready()
    runtime.validate_runtime_ready()


def test_runtime_validation_requires_embedding_provider(tmp_path: Path) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path / "worker",
        chat_api_key="chat-secret",
        feishu_app_id="cli-app",
        feishu_app_secret="secret",
        _env_file=None,
    )

    with pytest.raises(ValueError) as exc_info:
        settings.validate_runtime_ready()

    message = str(exc_info.value)
    assert "MEMOPILOT_EMBEDDING_BASE_URL" in message
    assert "MEMOPILOT_EMBEDDING_MODEL" in message
    assert "MEMOPILOT_EMBEDDING_API_KEY" in message
    assert "MEMOPILOT_EMBEDDING_DIMENSION" in message


def test_toml_loader_uses_environment_for_omitted_embedding_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMOPILOT_EMBEDDING_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("MEMOPILOT_EMBEDDING_MODEL", "env-embedding")
    monkeypatch.setenv("MEMOPILOT_EMBEDDING_API_KEY", "env-secret")
    monkeypatch.setenv("MEMOPILOT_EMBEDDING_DIMENSION", "768")
    config = tmp_path / "config.toml"
    config.write_text("[llm.main]\nmodel = 'deepseek-chat'\n", encoding="utf-8")

    settings = load_settings(config, workspace=tmp_path / "workspace")

    assert settings.embedding_base_url == "https://env.example/v1"
    assert settings.embedding_model == "env-embedding"
    assert settings.embedding_api_key.get_secret_value() == "env-secret"
    assert settings.embedding_dimension == 768


def test_toml_loader_parses_array_only_mcp_stdio_servers(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[[mcp.servers]]
server_id = "local"
command = ["python"]
args = ["server.py"]
env = { TOKEN = "${MCP_TOKEN}" }
startup_timeout_s = 3
call_timeout_s = 4
""",
        encoding="utf-8",
    )
    settings = load_settings(config, workspace=tmp_path / "workspace")

    assert len(settings.mcp_servers) == 1
    server = settings.mcp_servers[0]
    assert server.command == ("python",)
    assert server.args == ("server.py",)
    assert server.startup_timeout_seconds == 3
    assert server.call_timeout_seconds == 4


def test_toml_loader_does_not_shadow_env_for_omitted_memory_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMOPILOT_MEMORY_OPTIMIZER_ENABLED", "false")
    monkeypatch.setenv("MEMOPILOT_MEMORY_OPTIMIZER_INTERVAL_SECONDS", "3600")
    config = tmp_path / "config.toml"
    config.write_text("[llm.main]\nmodel = 'deepseek-chat'\n", encoding="utf-8")

    settings = load_settings(config, workspace=tmp_path / "workspace")

    assert settings.memory_optimizer_enabled is False
    assert settings.memory_optimizer_interval_seconds == 3600


def test_toml_loader_parses_proactive_settings(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[[mcp.servers]]
server_id = "feeds"
command = ["python"]
args = ["feeds.py"]

[proactive]
enabled = true
tick_seconds = 1800
context_probability = 0.3
active_start_hour = 8
active_end_hour = 23
drift_enabled = false
drift_min_interval_hours = 4

""",
        encoding="utf-8",
    )

    settings = load_settings(config, workspace=tmp_path / "workspace")

    assert settings.proactive_enabled is True
    assert settings.proactive_tick_seconds == 1800
    assert settings.proactive_context_probability == 0.3
    assert settings.proactive_active_start_hour == 8
    assert settings.proactive_active_end_hour == 23
    assert settings.drift_enabled is False
    assert settings.drift_min_interval_hours == 4
def test_proactive_enabled_without_sources_supports_drift_with_diagnostic(tmp_path: Path) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path,
        proactive_enabled=True,
        _env_file=None,
    )

    assert settings.proactive_diagnostics == (
        "主动唤醒已启用但未配置 Proactive Source；当前只可运行纯 Drift。",
    )


def test_fixed_proactive_tick_rejects_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMOPILOT_PROACTIVE_TICK_SECONDS", "600")
    config = tmp_path / "config.toml"
    config.write_text("[proactive]\nenabled = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="1800"):
        load_settings(config, workspace=tmp_path / "workspace")


def test_toml_mcp_server_rejects_literal_environment_secret(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[[mcp.servers]]
server_id = "feeds"
command = ["python"]
env = { TOKEN = "literal-secret" }
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\$\{ENV_NAME\}"):
        load_settings(config, workspace=tmp_path / "workspace")
