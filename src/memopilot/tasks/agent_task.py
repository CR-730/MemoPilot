"""领域任务合同。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AgentTask:
    task_id: str
    kind: str
    priority: int
    session_key: str
    payload: dict[str, object]
    created_at: datetime

    @property
    def payload_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True)


__all__ = ["AgentTask"]
