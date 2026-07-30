"""运行时会话历史端口实现。"""

from __future__ import annotations

from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.history import expand_history
from memopilot.tasks.operational import OperationalRepository


class OperationalSessionManager:
    def __init__(self, repository: OperationalRepository) -> None:
        self._repository = repository

    def get_history(self, session_key: str, limit: int) -> tuple[ChatMessage, ...]:
        return expand_history(self._repository.list_recent_messages(session_key, limit=limit))


__all__ = ["OperationalSessionManager"]
