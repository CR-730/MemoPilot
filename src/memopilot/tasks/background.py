"""不依赖运行审计表的后台任务载荷。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class BackgroundTask:
    """Scheduler 交给 Redis 的最小任务描述。

    task_id 只用于 Redis 幂等、日志和关联业务结果。
    """

    task_id: str
    kind: str
    priority: int
    session_key: str
    payload: Mapping[str, object]
    created_at: datetime

    @property
    def payload_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True)


__all__ = ["BackgroundTask"]
