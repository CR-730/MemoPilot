"""插件运行上下文与轻量 KV 存储。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memopilot.extensions.plugin_config import PluginConfig


@dataclass(slots=True)
class PluginContext:
    event_bus: Any
    tool_registry: Any
    plugin_id: str
    plugin_dir: Path
    kv_store: PluginKVStore
    config: PluginConfig | None = None
    workspace: Path | None = None
    session_manager: Any = None
    memory_engine: Any = None


class PluginKVStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        data = self._read()
        data[key] = value
        self._write(data)

    def increment(self, key: str, delta: int = 1) -> int:
        data = self._read()
        value = int(data.get(key, 0)) + delta
        data[key] = value
        self._write(data)
        return value

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"插件 KV 必须是 JSON 对象: {self._path}")
        return raw

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = ["PluginContext", "PluginKVStore"]
