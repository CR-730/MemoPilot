"""从会话仓储装载并展开模型历史消息。"""

from __future__ import annotations

from memopilot.persistence.conversation import ConversationRepository
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.history import expand_history


class OperationalSessionManager:
    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository

    def get_history(self, session_key: str, limit: int) -> tuple[ChatMessage, ...]:
        return expand_history(self._repository.list_recent_messages(session_key, limit=limit))


__all__ = ["OperationalSessionManager"]
