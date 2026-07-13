"""向量记忆数据库的 Embedding 身份校验。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from memopilot.persistence.migrations import connect_database, execute_with_busy_retry


class EmbeddingConfigurationMismatchError(RuntimeError):
    """当前 Embedding Provider 与已有向量数据库不兼容。"""


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    base_url: str
    model: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip() or self.dimension <= 0:
            raise ValueError("Embedding identity 必须包含 base_url、model 和正向量维度")


def ensure_embedding_identity(
    database: Path,
    identity: EmbeddingIdentity,
    *,
    busy_timeout_seconds: float = 5,
    now: datetime | None = None,
) -> None:
    """首次写入 Provider 身份；后续启动只接受完全相同的身份。"""
    expected = json.dumps(asdict(identity), ensure_ascii=False, sort_keys=True)
    timestamp = (now or datetime.now(UTC)).isoformat()
    connection = connect_database(database, busy_timeout_seconds=busy_timeout_seconds)
    try:

        def check_and_store() -> None:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT value_json FROM memory_metadata WHERE key = ?",
                    ("embedding_identity",),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO memory_metadata(key, value_json, updated_at) VALUES (?, ?, ?)",
                        ("embedding_identity", expected, timestamp),
                    )
                elif str(row[0]) != expected:
                    actual = json.loads(str(row[0]))
                    raise EmbeddingConfigurationMismatchError(
                        f"Embedding 配置与 memory2.db 不匹配: expected={asdict(identity)}, "
                        f"actual={actual}"
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

        execute_with_busy_retry(check_and_store)
    finally:
        connection.close()


__all__ = [
    "EmbeddingConfigurationMismatchError",
    "EmbeddingIdentity",
    "ensure_embedding_identity",
]
