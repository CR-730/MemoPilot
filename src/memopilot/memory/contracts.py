"""记忆引擎、检索与模型适配之间的稳定合同。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

MemoryIntent = Literal["context", "answer", "timeline", "interest", "procedure"]


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    text: str
    intent: MemoryIntent = "answer"
    session_key: str = ""
    memory_kinds: tuple[str, ...] = ()
    time_start: datetime | None = None
    time_end: datetime | None = None
    limit: int = 8


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    source_ref: str
    kind: str = "message_range"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    kind: str
    summary: str
    score: float
    evidence: tuple[MemoryEvidence, ...] = ()
    signals: dict[str, object] = field(default_factory=dict)
    injected: bool = False


@dataclass(frozen=True, slots=True)
class MemoryQueryResult:
    text_block: str = ""
    records: tuple[MemoryRecord, ...] = ()
    trace: dict[str, object] = field(default_factory=dict)


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class HypothesisProvider(Protocol):
    async def generate(self, query: str, *, style: str) -> str: ...


class MemoryQueryEngine(Protocol):
    async def query(self, request: MemoryQuery) -> MemoryQueryResult: ...


__all__ = [
    "EmbeddingProvider",
    "HypothesisProvider",
    "MemoryEvidence",
    "MemoryIntent",
    "MemoryQuery",
    "MemoryQueryEngine",
    "MemoryQueryResult",
    "MemoryRecord",
]
