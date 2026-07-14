from __future__ import annotations

from datetime import UTC, datetime

from memopilot.memory.contracts import MemoryQueryResult, MemoryRecord
from memopilot.memory.tools import build_recall_memory_tool
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import ToolRegistry


class _Engine:
    def __init__(self) -> None:
        self.requests = []

    async def query(self, request):
        self.requests.append(request)
        return MemoryQueryResult(
            records=(MemoryRecord("m1", "event", "阶段四完成", 0.8),),
            trace={"intent": request.intent},
        )


async def test_recall_memory_tool_exposes_intents_and_structured_evidence() -> None:
    engine = _Engine()
    registry = ToolRegistry([build_recall_memory_tool(engine)])  # type: ignore[arg-type]

    observation = await registry.execute(
        FunctionCall(
            id="call-1",
            name="recall_memory",
            arguments={"query": "做完了什么", "intent": "answer", "limit": 4},
        )
    )

    assert observation.ok is True
    assert engine.requests[0].text == "做完了什么"
    assert engine.requests[0].intent == "answer"
    assert observation.result["records"][0]["id"] == "m1"


async def test_recall_memory_tool_parses_timeline_boundaries() -> None:
    engine = _Engine()
    registry = ToolRegistry([build_recall_memory_tool(engine)])  # type: ignore[arg-type]

    observation = await registry.execute(
        FunctionCall(
            id="call-2",
            name="recall_memory",
            arguments={
                "query": "阶段记录",
                "intent": "timeline",
                "time_start": "2026-07-01T00:00:00+00:00",
                "time_end": "2026-08-01T00:00:00+00:00",
            },
        )
    )

    assert observation.ok is True
    assert engine.requests[0].time_start == datetime(2026, 7, 1, tzinfo=UTC)
    assert engine.requests[0].time_end == datetime(2026, 8, 1, tzinfo=UTC)
