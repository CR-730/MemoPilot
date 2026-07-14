"""Channel 共用的附件、去重与身份索引组件。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from memopilot.tasks.operational import SessionIdentityRecord


class IdentityStore(Protocol):
    def list_session_identities(self, channel: str) -> tuple[SessionIdentityRecord, ...]: ...

    def remember_session_identities(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        identities: Mapping[str, str],
        now: datetime,
    ) -> None: ...


class AttachmentStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write_bytes(self, data: bytes, *, prefix: str, suffix: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{prefix}{uuid4().hex}{suffix}"
        path.write_bytes(data)
        return path


class MessageDeduper:
    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._seen: set[str] = set()
        self._in_flight: set[str] = set()
        self._order: deque[str] = deque()

    def seen(self, key: str) -> bool:
        if not self.reserve(key):
            return True
        self.commit(key)
        return False

    def reserve(self, key: str) -> bool:
        if key in self._seen or key in self._in_flight:
            return False
        self._in_flight.add(key)
        return True

    def commit(self, key: str) -> None:
        self._in_flight.discard(key)
        if key in self._seen:
            return
        self._seen.add(key)
        self._order.append(key)
        while len(self._order) > self._max_size:
            self._seen.discard(self._order.popleft())

    def release(self, key: str) -> None:
        self._in_flight.discard(key)


class SessionIdentityIndex:
    def __init__(
        self,
        store: IdentityStore,
        *,
        channel: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._channel = channel
        self._clock = clock or (lambda: datetime.now(UTC))
        self.mapping: dict[str, str] = {}

    def rebuild(self) -> dict[str, str]:
        self.mapping = {
            record.identity_value: record.chat_id
            for record in self._store.list_session_identities(self._channel)
        }
        return dict(self.mapping)

    def resolve(self, identity: str) -> str | None:
        return self.mapping.get(str(identity).strip())

    def remember(
        self,
        *,
        session_key: str,
        chat_id: str,
        identities: Mapping[str, str],
    ) -> None:
        normalized = {kind: str(value).strip() for kind, value in identities.items() if value}
        self._store.remember_session_identities(
            session_key=session_key,
            channel=self._channel,
            chat_id=chat_id,
            identities=normalized,
            now=self._clock(),
        )
        self.mapping.update({value: chat_id for value in normalized.values()})


__all__: Sequence[str] = ("AttachmentStore", "MessageDeduper", "SessionIdentityIndex")
