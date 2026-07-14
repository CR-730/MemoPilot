"""记忆工具的显式 Function Calling 合同。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from memopilot.memory.contracts import MemoryQuery, MemoryQueryEngine
from memopilot.runtime.tools import Tool


def build_recall_memory_tool(engine: MemoryQueryEngine) -> Tool:
    async def recall_memory(
        *,
        query: str,
        intent: str = "answer",
        limit: int = 8,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> dict[str, object]:
        result = await engine.query(
            MemoryQuery(
                text=query,
                intent=intent,  # type: ignore[arg-type]
                limit=limit,
                time_start=_parse_time(time_start),
                time_end=_parse_time(time_end),
            )
        )
        return {
            "text_block": result.text_block,
            "records": [asdict(record) for record in result.records],
            "trace": result.trace,
        }

    return Tool(
        name="recall_memory",
        description=(
            "检索长期记忆。回答问题时用 answer；按时间回顾用 timeline；"
            "查偏好用 interest；查既有流程用 procedure。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "intent": {
                    "type": "string",
                    "enum": ["answer", "timeline", "interest", "procedure"],
                    "default": "answer",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 8},
                "time_start": {"type": "string", "format": "date-time"},
                "time_end": {"type": "string", "format": "date-time"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=recall_memory,
    )


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


__all__ = ["build_recall_memory_tool"]
