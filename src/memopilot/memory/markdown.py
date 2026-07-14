"""可恢复的 Markdown 长期记忆文件存储。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_ALLOWED_FILES = frozenset(
    {"MEMORY.md", "SELF.md", "HISTORY.md", "PENDING.md", "CONTEXT.md"}
)


class MarkdownMemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot_path(self) -> Path:
        return self.root / ".PENDING.snapshot.md"

    def read(self, name: str) -> str:
        path = self._path(name)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def replace(self, name: str, content: str) -> None:
        self._atomic_replace(self._path(name), content.strip() + ("\n" if content.strip() else ""))

    def append(self, name: str, content: str) -> None:
        value = content.strip()
        if not value:
            return
        current = self.read(name).rstrip()
        merged = f"{current}\n\n{value}\n" if current else f"{value}\n"
        self._atomic_replace(self._path(name), merged)

    def append_artifact(
        self,
        name: str,
        *,
        consolidation_id: str,
        content: str,
        content_hash: str,
    ) -> None:
        if self.contains_artifact(name, consolidation_id, content_hash):
            return
        marker = self._marker(consolidation_id, content_hash)
        self.append(name, f"{marker}\n{content.strip()}")

    def contains_artifact(self, name: str, consolidation_id: str, content_hash: str) -> bool:
        return self._marker(consolidation_id, content_hash) in self.read(name)

    def begin_pending_snapshot(self) -> str:
        if self.snapshot_path.exists():
            raise RuntimeError("PENDING 快照已存在，必须先恢复")
        pending = self._path("PENDING.md")
        if pending.exists():
            os.replace(pending, self.snapshot_path)
        else:
            self._atomic_replace(self.snapshot_path, "")
        self._atomic_replace(pending, "")
        return self.snapshot_path.read_text(encoding="utf-8")

    def commit_pending_snapshot(self) -> None:
        self.snapshot_path.unlink(missing_ok=True)

    def rollback_pending_snapshot(self) -> bool:
        if not self.snapshot_path.exists():
            return False
        snapshot = self.snapshot_path.read_text(encoding="utf-8").strip()
        current = self.read("PENDING.md").strip()
        merged = "\n\n".join(value for value in (snapshot, current) if value)
        self.replace("PENDING.md", merged)
        self.snapshot_path.unlink(missing_ok=True)
        return True

    def _path(self, name: str) -> Path:
        if name not in _ALLOWED_FILES:
            raise ValueError(f"不支持的记忆文件: {name}")
        return self.root / name

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _marker(consolidation_id: str, content_hash: str) -> str:
        return f"<!-- memopilot:{consolidation_id}:{content_hash} -->"

    @staticmethod
    def _atomic_replace(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)


__all__ = ["MarkdownMemoryStore"]
