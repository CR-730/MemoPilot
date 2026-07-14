"""SQLite 连接策略与显式 schema 迁移。"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memopilot.config import MemoPilotSettings


class DatabaseKind(StrEnum):
    OPERATIONAL = "operational"
    MEMORY = "memory2"
    WAKE = "wake"


class MigrationError(RuntimeError):
    """数据库迁移未能原子完成。"""


class UnsupportedDatabaseVersionError(MigrationError):
    """数据库版本高于当前程序能够理解的版本。"""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


@dataclass(frozen=True, slots=True)
class MigrationReport:
    database: Path
    kind: DatabaseKind
    from_version: int
    to_version: int
    applied_versions: tuple[int, ...]
    backup_path: Path | None


def connect_database(
    database: Path,
    *,
    busy_timeout_seconds: float = 5,
) -> sqlite3.Connection:
    """按 MemoPilot 的统一 PRAGMA 打开 SQLite 连接。"""
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        database,
        timeout=busy_timeout_seconds,
        isolation_level=None,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1000)}")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def execute_with_busy_retry[T](
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.05,
    sleep: Callable[[float], object] = time.sleep,
) -> T:
    """只对 SQLite busy/locked 做有界指数退避。"""
    if max_attempts < 1:
        raise ValueError("max_attempts 必须至少为 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            retryable = "locked" in message or "busy" in message
            if not retryable or attempt == max_attempts:
                raise
            sleep(base_delay_seconds * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def migrate_database(
    database: Path,
    kind: DatabaseKind,
    *,
    migrations: Sequence[Migration] | None = None,
    busy_timeout_seconds: float = 5,
) -> MigrationReport:
    """把单个数据库迁移到当前支持版本。"""
    database = Path(database)
    existed = database.exists() and database.stat().st_size > 0
    selected = tuple(migrations) if migrations is not None else _load_migrations(kind)
    _validate_migration_sequence(selected)
    target_version = selected[-1].version if selected else 0
    connection = connect_database(database, busy_timeout_seconds=busy_timeout_seconds)
    try:
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current_version > target_version:
            raise UnsupportedDatabaseVersionError(
                f"{database.name} schema 版本 {current_version} 高于当前支持版本 "
                f"{target_version}，拒绝降级启动"
            )

        pending = tuple(item for item in selected if item.version > current_version)
        backup_path = (
            _backup_database(connection, database, current_version)
            if existed and pending
            else None
        )
        applied: list[int] = []
        for migration in pending:
            _apply_migration(connection, migration)
            applied.append(migration.version)
        return MigrationReport(
            database=database,
            kind=kind,
            from_version=current_version,
            to_version=applied[-1] if applied else current_version,
            applied_versions=tuple(applied),
            backup_path=backup_path,
        )
    finally:
        connection.close()


def migrate_all_databases(settings: MemoPilotSettings) -> tuple[MigrationReport, ...]:
    """按固定名称初始化 operational、memory2 与 wake 三个数据库。"""
    timeout = settings.sqlite_busy_timeout_seconds
    return (
        migrate_database(
            settings.operational_database,
            DatabaseKind.OPERATIONAL,
            busy_timeout_seconds=timeout,
        ),
        migrate_database(
            settings.memory_database,
            DatabaseKind.MEMORY,
            busy_timeout_seconds=timeout,
        ),
        migrate_database(
            settings.wake_database,
            DatabaseKind.WAKE,
            busy_timeout_seconds=timeout,
        ),
    )


def _load_migrations(kind: DatabaseKind) -> tuple[Migration, ...]:
    schema = files("memopilot.persistence.schema")
    versions = (1, 2) if kind is DatabaseKind.OPERATIONAL else (1,)
    return tuple(
        Migration(
            version=version,
            name=f"migrate_{kind.value}_v{version}",
            sql=schema.joinpath(f"{kind.value}_v{version}.sql").read_text("utf-8"),
        )
        for version in versions
    )


def _validate_migration_sequence(migrations: Sequence[Migration]) -> None:
    versions = [item.version for item in migrations]
    if versions != sorted(set(versions)) or any(version < 1 for version in versions):
        raise ValueError("迁移版本必须从正整数开始并严格递增且不重复")


def _backup_database(
    connection: sqlite3.Connection,
    database: Path,
    current_version: int,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = database.with_name(
        f"{database.name}.bak-v{current_version}-{timestamp}"
    )
    with sqlite3.connect(backup_path) as backup:
        connection.backup(backup)
    return backup_path


def _apply_migration(connection: sqlite3.Connection, migration: Migration) -> None:
    def apply_once() -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _iter_sql_statements(migration.sql):
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {migration.version}")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    try:
        execute_with_busy_retry(apply_once)
    except Exception as exc:
        raise MigrationError(
            f"迁移 {migration.version} ({migration.name}) 执行失败: {exc}"
        ) from exc


def _iter_sql_statements(script: str) -> Iterator[str]:
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            buffer = ""
            if statement:
                yield statement
    if buffer.strip():
        raise MigrationError("迁移 SQL 包含未结束的语句")


__all__ = [
    "DatabaseKind",
    "Migration",
    "MigrationError",
    "MigrationReport",
    "UnsupportedDatabaseVersionError",
    "connect_database",
    "execute_with_busy_retry",
    "migrate_all_databases",
    "migrate_database",
]
