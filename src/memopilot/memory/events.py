from __future__ import annotations

from dataclasses import dataclass

from memopilot.memory.contracts import MemoryRecord


@dataclass(frozen=True, slots=True)
class RetrievalCompleted:
    session_key: str
    query: str
    intent: str
    records: tuple[MemoryRecord, ...]
    aux_queries: tuple[str, ...]
    injected_count: int


@dataclass(frozen=True, slots=True)
class MemoryWritten:
    session_key: str
    source_ref: str
    memory_type: str
    item_id: str
    status: str
    summary: str


__all__ = ["MemoryWritten", "RetrievalCompleted"]
