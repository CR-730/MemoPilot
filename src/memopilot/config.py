"""MemoPilot 的类型化配置与启动前校验。"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class MemoPilotSettings(BaseSettings):
    """运行配置。

    初始化参数主要服务于测试和显式嵌入；常规运行的配置优先级为环境变量、
    ``.env``、``config/default.yaml``、字段默认值。
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        yaml_file=Path("config/default.yaml"),
        yaml_file_encoding="utf-8",
    )

    workspace: Path = Path("workspace")
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

    wake_tick_seconds: int = Field(default=300, gt=0)
    content_half_life_hours: float = Field(default=6, gt=0)
    proactive_cooldown_hours: float = Field(default=2, ge=0)
    lease_ttl_seconds: int = Field(default=30, gt=0)
    lease_heartbeat_seconds: int = Field(default=10, gt=0)
    reclaim_idle_seconds: int = Field(default=60, gt=0)
    sqlite_busy_timeout_seconds: float = Field(default=5, gt=0)
    mcp_startup_timeout_seconds: float = Field(default=15, gt=0)
    mcp_call_timeout_seconds: float = Field(default=30, gt=0)
    llm_retry_limit: int = Field(default=2, ge=0)
    llm_max_iterations: int = Field(default=10, gt=0)
    llm_max_output_tokens: int = Field(default=2048, gt=0)
    llm_timeout_seconds: float = Field(default=60, gt=0)
    llm_thinking_enabled: bool = False
    memory_short_term_message_limit: int = Field(default=12, gt=0)
    memory_consolidation_keep_count: int = Field(default=12, ge=0)
    memory_consolidation_min_new_messages: int = Field(default=5, gt=0)
    memory_retrieval_limit: int = Field(default=8, gt=0, le=200)
    memory_score_threshold: float = Field(default=0.45, ge=0, le=1)
    memory_relative_delta: float = Field(default=0.06, ge=0, le=1)
    memory_inject_max_chars: int = Field(default=1200, ge=120)
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
            YamlConfigSettingsSource(settings_cls),
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

    @model_validator(mode="after")
    def _resolve_bounded_paths(self) -> Self:
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
    def operational_database(self) -> Path:
        return self.data_dir / "operational.db"

    @property
    def memory_database(self) -> Path:
        return self.data_dir / "memory2.db"

    @property
    def wake_database(self) -> Path:
        return self.data_dir / "wake.db"

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

    def validate_worker_ready(self) -> None:
        missing = []
        if not self.chat_api_key.get_secret_value():
            missing.append("MEMOPILOT_CHAT_API_KEY")
        if not self.embedding_base_url:
            missing.append("MEMOPILOT_EMBEDDING_BASE_URL")
        if not self.embedding_model:
            missing.append("MEMOPILOT_EMBEDDING_MODEL")
        if not self.embedding_api_key.get_secret_value():
            missing.append("MEMOPILOT_EMBEDDING_API_KEY")
        if not self.embedding_dimension:
            missing.append("MEMOPILOT_EMBEDDING_DIMENSION")
        if not self.feishu_app_id:
            missing.append("MEMOPILOT_FEISHU_APP_ID")
        if not self.feishu_app_secret.get_secret_value():
            missing.append("MEMOPILOT_FEISHU_APP_SECRET")
        if missing:
            raise ValueError("Worker 启动配置不完整，缺少: " + ", ".join(missing))

    def validate_effects_ready(self) -> None:
        """Effects 进程只校验自身发送飞书消息需要的配置。"""
        missing = []
        if not self.feishu_app_id:
            missing.append("MEMOPILOT_FEISHU_APP_ID")
        if not self.feishu_app_secret.get_secret_value():
            missing.append("MEMOPILOT_FEISHU_APP_SECRET")
        if missing:
            raise ValueError("Effects 启动配置不完整，缺少: " + ", ".join(missing))


def load_settings(
    config_path: str | Path,
    *,
    workspace: Path | None = None,
) -> MemoPilotSettings:
    """按原型 TOML 字段加载阶段 3 所需配置。"""
    path = Path(config_path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"主配置仅支持 TOML: {path.suffix}")
    data = _resolve_environment(tomllib.loads(path.read_text(encoding="utf-8")))
    llm = _as_dict(data.get("llm"))
    main = _as_dict(llm.get("main"))
    agent = _as_dict(data.get("agent"))
    memory = _as_dict(data.get("memory"))
    embedding = _as_dict(memory.get("embedding"))
    channels = _as_dict(data.get("channels"))
    feishu = _as_dict(channels.get("feishu"))
    redis = _as_dict(data.get("redis"))
    values: dict[str, Any] = {
        "chat_model": main.get("model") or "deepseek-v4-flash",
        "chat_base_url": main.get("base_url") or "https://api.deepseek.com",
        "llm_thinking_enabled": bool(main.get("enable_thinking", False)),
        "llm_max_output_tokens": int(agent.get("max_tokens", 2048)),
        "llm_max_iterations": int(agent.get("max_iterations", 10)),
        "feishu_enabled": bool(feishu.get("enabled", True)),
        "feishu_allow_from": feishu.get("allow_from", feishu.get("allowFrom", ())),
        "feishu_channel_name": feishu.get("channel_name", "feishu"),
        "feishu_receive_mode": feishu.get("receive_mode", "ws"),
        "redis_url": redis.get("url") or data.get("redis_url") or "redis://localhost:6379/0",
    }
    optional_values = {
        "chat_api_key": main.get("api_key"),
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
    return MemoPilotSettings(**values, _env_file=None)  # type: ignore[call-arg]


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


__all__ = ["MemoPilotSettings", "load_settings"]
