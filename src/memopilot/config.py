"""MemoPilot 的类型化配置与启动前校验。"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Self

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from memopilot.extensions.mcp import McpServerConfig


class MemoPilotSettings(BaseSettings):
    """运行配置。

    初始化参数主要服务于测试和显式嵌入；常规运行的配置优先级为环境变量、
    ``.env``、字段默认值。
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    workspace: Path = Field(
        default_factory=lambda: Path.home() / ".memopilot" / "memopilot-workspace"
    )
    data_dir: Path = Path("data")
    memory_dir: Path = Path("memory")
    journal_dir: Path = Path("journal")
    uploads_dir: Path = Path("uploads")
    restore_dir: Path = Path("restore")
    traces_dir: Path = Path("traces")
    plugins_dir: Path = Path("plugins")
    skills_dir: Path = Path("skills")

    chat_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-v4-flash"
    chat_api_key: SecretStr = SecretStr("")
    chat_multimodal: bool = True

    fast_base_url: str = ""
    fast_model: str = ""
    fast_api_key: SecretStr = SecretStr("")

    vl_base_url: str = ""
    vl_model: str = ""
    vl_api_key: SecretStr = SecretStr("")

    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_dimension: int = Field(default=0, ge=0)

    redis_url: str = "redis://localhost:6379/0"

    feishu_app_id: str = ""
    feishu_app_secret: SecretStr = SecretStr("")
    feishu_allow_from: Annotated[tuple[str, ...], NoDecode] = ()
    feishu_channel_name: str = "feishu"
    feishu_enabled: bool = True
    feishu_receive_mode: str = "ws"

    proactive_enabled: bool = False
    proactive_tick_seconds: int = Field(default=1800, ge=1800, le=1800)
    proactive_context_probability: float = Field(default=0.3, ge=0, le=1)
    proactive_active_start_hour: int = Field(default=8, ge=0, le=23)
    proactive_active_end_hour: int = Field(default=23, ge=1, le=24)
    drift_enabled: bool = True
    drift_min_interval_hours: float = Field(default=3, ge=0)
    sqlite_busy_timeout_seconds: float = Field(default=5, gt=0)
    mcp_startup_timeout_seconds: float = Field(default=15, gt=0)
    mcp_call_timeout_seconds: float = Field(default=30, gt=0)
    mcp_servers: Annotated[tuple[McpServerConfig, ...], NoDecode] = ()
    llm_retry_limit: int = Field(default=2, ge=0)
    llm_max_iterations: int = Field(default=10, gt=0)
    llm_max_output_tokens: int = Field(default=2048, gt=0)
    llm_context_window_tokens: int = Field(default=1_000_000, gt=0)
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_thinking_enabled: bool = False
    tool_search_enabled: bool = True
    memory_window: int = Field(default=40, gt=0)
    memory_retrieval_limit: int = Field(default=8, gt=0, le=200)
    memory_score_threshold: float = Field(default=0.45, ge=0, le=1)
    memory_score_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "procedure": 0.58,
            "preference": 0.52,
            "event": 0.45,
            "profile": 0.5,
        }
    )
    memory_embed_timeout_seconds: float = Field(default=5, gt=0)
    memory_procedure_guard_enabled: bool = True
    memory_hotness_alpha: float = Field(default=0.2, ge=0, le=1)
    memory_hotness_half_life_days: float = Field(default=14, gt=0)
    memory_inject_max_chars: int = Field(default=1200, ge=120)
    memory_inject_max_forced: int = Field(default=3, gt=0)
    memory_inject_max_procedure_preference: int = Field(default=4, gt=0)
    memory_inject_max_event_profile: int = Field(default=2, ge=0)
    memory_optimizer_enabled: bool = True
    memory_optimizer_interval_seconds: int = Field(default=64800, gt=0)
    display_timezone: str = "Asia/Shanghai"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """在 Pydantic 默认源中加入低优先级 YAML 默认配置。"""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @field_validator("feishu_allow_from", mode="before")
    @classmethod
    def _parse_allowlist(cls, value: Any) -> tuple[str, ...]:
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                value = json.loads(text)
            else:
                value = text.split(",")
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("飞书 owner allowlist 必须是 ID 列表或逗号分隔字符串")
        return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    @field_validator("redis_url")
    @classmethod
    def _normalize_local_redis_host(cls, value: str) -> str:
        return re.sub(
            r"^(rediss?://)localhost(?=[:/]|$)",
            r"\g<1>127.0.0.1",
            value,
            count=1,
            flags=re.IGNORECASE,
        )

    @model_validator(mode="after")
    def _resolve_bounded_paths(self) -> Self:
        if self.proactive_active_start_hour >= self.proactive_active_end_hour:
            raise ValueError("主动发送时段必须满足 start < end")
        workspace = self.workspace.expanduser().resolve()
        self.workspace = workspace
        for field_name in (
            "data_dir",
            "memory_dir",
            "journal_dir",
            "uploads_dir",
            "restore_dir",
            "traces_dir",
            "plugins_dir",
            "skills_dir",
        ):
            configured = getattr(self, field_name).expanduser()
            resolved = (
                configured.resolve()
                if configured.is_absolute()
                else (workspace / configured).resolve()
            )
            try:
                resolved.relative_to(workspace)
            except ValueError as exc:
                raise ValueError(f"{field_name} 必须位于 workspace 内: {resolved}") from exc
            setattr(self, field_name, resolved)
        return self

    @property
    def proactive_diagnostics(self) -> tuple[str, ...]:
        if self.proactive_enabled and not (self.workspace / "proactive_sources.json").exists():
            return ("主动唤醒已启用但未配置 Proactive Source；当前只可运行纯 Drift。",)
        return ()

    @property
    def memory_history_limit(self) -> int:
        return max(4, (self.memory_window + 3) // 4 * 4) // 2

    @property
    def memory_consolidation_keep_count(self) -> int:
        return self.memory_history_limit

    @property
    def memory_consolidation_min_new_messages(self) -> int:
        return max(5, self.memory_history_limit // 2)

    @property
    def memory_recent_turn_count(self) -> int:
        return max(1, self.memory_history_limit // 2)

    @property
    def operational_database(self) -> Path:
        return self.data_dir / "operational.db"

    @property
    def memory_database(self) -> Path:
        return self.data_dir / "memory2.db"

    @property
    def proactive_database(self) -> Path:
        return self.data_dir / "proactive.db"

    def validate_runtime_ready(self) -> None:
        """聚合报告启动核心链路缺少的配置。"""
        required = {
            "MEMOPILOT_CHAT_API_KEY": self.chat_api_key.get_secret_value(),
            "MEMOPILOT_EMBEDDING_BASE_URL": self.embedding_base_url,
            "MEMOPILOT_EMBEDDING_MODEL": self.embedding_model,
            "MEMOPILOT_EMBEDDING_API_KEY": self.embedding_api_key.get_secret_value(),
            "MEMOPILOT_EMBEDDING_DIMENSION": self.embedding_dimension,
            "MEMOPILOT_FEISHU_APP_ID": self.feishu_app_id,
            "MEMOPILOT_FEISHU_APP_SECRET": self.feishu_app_secret.get_secret_value(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("启动配置不完整，缺少: " + ", ".join(missing))

    def validate_app_ready(self) -> None:
        missing: list[str] = []
        if not self.feishu_enabled:
            missing.append("channels.feishu.enabled=true")
        if not self.feishu_app_id:
            missing.append("MEMOPILOT_FEISHU_APP_ID")
        if not self.feishu_app_secret.get_secret_value():
            missing.append("MEMOPILOT_FEISHU_APP_SECRET")
        if self.feishu_receive_mode.lower() != "ws":
            missing.append("channels.feishu.receive_mode=ws")
        if missing:
            raise ValueError("App 启动配置不完整，缺少或不支持: " + ", ".join(missing))

def load_settings(
    config_path: str | Path,
    *,
    workspace: Path | None = None,
) -> MemoPilotSettings:
    """按原型 TOML 字段加载阶段 3 所需配置。"""
    path = Path(config_path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"主配置仅支持 TOML: {path.suffix}")
    load_dotenv(path.parent / ".env", override=False)
    raw_data = tomllib.loads(path.read_text(encoding="utf-8"))
    data = _resolve_environment(raw_data)
    llm = _as_dict(data.get("llm"))
    main = _as_dict(llm.get("main"))
    fast = _as_dict(llm.get("fast"))
    vl = _as_dict(llm.get("vl"))
    agent = _as_dict(data.get("agent"))
    agent_tools = _as_dict(agent.get("tools"))
    memory = _as_dict(data.get("memory"))
    embedding = _as_dict(memory.get("embedding"))
    channels = _as_dict(data.get("channels"))
    feishu = _as_dict(channels.get("feishu"))
    redis = _as_dict(data.get("redis"))
    raw_mcp = _as_dict(raw_data.get("mcp"))
    proactive = _as_dict(data.get("proactive"))
    values: dict[str, Any] = {}
    explicit_fields = (
        (main, "model", "chat_model"),
        (main, "base_url", "chat_base_url"),
        (main, "enable_thinking", "llm_thinking_enabled"),
        (main, "multimodal", "chat_multimodal"),
        (fast, "model", "fast_model"),
        (fast, "base_url", "fast_base_url"),
        (vl, "model", "vl_model"),
        (vl, "base_url", "vl_base_url"),
        (agent, "max_tokens", "llm_max_output_tokens"),
        (agent, "max_iterations", "llm_max_iterations"),
        (agent, "context_window_tokens", "llm_context_window_tokens"),
        (agent_tools, "search_enabled", "tool_search_enabled"),
        (feishu, "enabled", "feishu_enabled"),
        (feishu, "channel_name", "feishu_channel_name"),
        (feishu, "receive_mode", "feishu_receive_mode"),
        (memory, "score_threshold", "memory_score_threshold"),
        (memory, "window", "memory_window"),
        (memory, "score_thresholds", "memory_score_thresholds"),
        (memory, "embed_timeout_seconds", "memory_embed_timeout_seconds"),
        (memory, "procedure_guard_enabled", "memory_procedure_guard_enabled"),
        (memory, "inject_max_forced", "memory_inject_max_forced"),
        (
            memory,
            "inject_max_procedure_preference",
            "memory_inject_max_procedure_preference",
        ),
        (memory, "inject_max_event_profile", "memory_inject_max_event_profile"),
        (memory, "hotness_alpha", "memory_hotness_alpha"),
        (memory, "hotness_half_life_days", "memory_hotness_half_life_days"),
        (memory, "optimizer_enabled", "memory_optimizer_enabled"),
        (
            memory,
            "optimizer_interval_seconds",
            "memory_optimizer_interval_seconds",
        ),
        (proactive, "enabled", "proactive_enabled"),
        (proactive, "tick_seconds", "proactive_tick_seconds"),
        (
            proactive,
            "context_probability",
            "proactive_context_probability",
        ),
        (proactive, "active_start_hour", "proactive_active_start_hour"),
        (proactive, "active_end_hour", "proactive_active_end_hour"),
        (proactive, "drift_enabled", "drift_enabled"),
        (proactive, "drift_min_interval_hours", "drift_min_interval_hours"),
    )
    for source, source_key, field_name in explicit_fields:
        if source_key in source:
            values[field_name] = source[source_key]
    if "allow_from" in feishu or "allowFrom" in feishu:
        values["feishu_allow_from"] = feishu.get("allow_from", feishu.get("allowFrom"))
    if "url" in redis:
        values["redis_url"] = redis["url"]
    elif "redis_url" in data:
        values["redis_url"] = data["redis_url"]
    if "servers" in raw_mcp:
        values["mcp_servers"] = _parse_mcp_servers(
            raw_mcp["servers"],
            workspace=workspace or Path("workspace"),
        )
    optional_values = {
        "chat_api_key": main.get("api_key"),
        "fast_api_key": fast.get("api_key"),
        "vl_api_key": vl.get("api_key"),
        "embedding_model": embedding.get("model"),
        "embedding_api_key": embedding.get("api_key"),
        "embedding_base_url": embedding.get("base_url"),
        "embedding_dimension": embedding.get("dimension"),
        "feishu_app_id": feishu.get("app_id", feishu.get("appId")),
        "feishu_app_secret": feishu.get("app_secret", feishu.get("appSecret")),
    }
    values.update({key: value for key, value in optional_values.items() if value not in (None, "")})
    if workspace is not None:
        values["workspace"] = workspace
    return MemoPilotSettings(**values)


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    if isinstance(value, str):
        resolved = re.sub(
            r"\$\{(\w+)\}",
            lambda match: os.environ.get(match.group(1), match.group(0)),
            value,
        )
        if re.fullmatch(r"\$\{\w+\}", resolved):
            return ""
        return resolved
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_mcp_servers(value: Any, *, workspace: Path) -> tuple[McpServerConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("mcp.servers 必须是 TOML 对象数组")
    servers: list[McpServerConfig] = []
    workspace_root = workspace.expanduser().resolve()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("mcp.servers 每项必须是对象")
        command = raw.get("command")
        args = raw.get("args", [])
        env = raw.get("env", {})
        if not isinstance(command, list):
            raise ValueError("MCP command 必须是字符串数组")
        if not isinstance(args, list):
            raise ValueError("MCP args 必须是字符串数组")
        if not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in env.items()
        ):
            raise ValueError("MCP env 必须是字符串映射")
        cwd_value = raw.get("cwd")
        cwd = None
        if cwd_value is not None:
            configured = Path(str(cwd_value)).expanduser()
            cwd = (
                configured.resolve()
                if configured.is_absolute()
                else (workspace_root / configured).resolve()
            )
            try:
                cwd.relative_to(workspace_root)
            except ValueError as exc:
                raise ValueError("MCP cwd 必须位于 workspace 内") from exc
        servers.append(
            McpServerConfig(
                server_id=str(raw.get("server_id") or ""),
                command=tuple(command),
                args=tuple(args),
                env={str(key): str(item) for key, item in env.items()},
                cwd=cwd,
                enabled=bool(raw.get("enabled", True)),
                startup_timeout_seconds=float(raw.get("startup_timeout_s", 15)),
                call_timeout_seconds=float(raw.get("call_timeout_s", 30)),
                shutdown_timeout_seconds=float(raw.get("shutdown_timeout_s", 5)),
                max_restarts=int(raw.get("max_restarts", 3)),
                model_result_max_chars=int(raw.get("model_result_max_chars", 12_000)),
            )
        )
    return tuple(servers)


__all__ = ["MemoPilotSettings", "load_settings"]
