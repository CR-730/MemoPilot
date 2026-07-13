"""MemoPilot 的类型化配置与启动前校验。"""

from __future__ import annotations

import json
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

    chat_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-chat"
    chat_api_key: SecretStr = SecretStr("")

    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_dimension: int = Field(default=0, ge=0)

    redis_url: str = "redis://localhost:6379/0"
    redis_stream_maxlen: int = Field(default=10_000, gt=0)

    feishu_app_id: str = ""
    feishu_app_secret: SecretStr = SecretStr("")
    feishu_allow_from: Annotated[tuple[str, ...], NoDecode] = ()
    feishu_channel_name: str = "feishu"

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
            "MEMOPILOT_FEISHU_ALLOW_FROM": self.feishu_allow_from,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("启动配置不完整，缺少: " + ", ".join(missing))


__all__ = ["MemoPilotSettings"]
