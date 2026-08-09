from datetime import UTC, datetime

import pytest

from memopilot.proactive.drift_executor import DriftTurnPipeline
from memopilot.runtime.contracts import FunctionCall, ModelResponse
from memopilot.runtime.tools import Tool, ToolExecutor, ToolRegistry

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


class _Repository:
    def get_activity_version(self, session_key: str) -> int:
        del session_key
        return 4

    def commit_direct_assistant(self, **_kwargs):  # type: ignore[no-untyped-def]
        return None


class _Selector:
    async def select(self) -> str:
        return "research"


class _Provider:
    def __init__(self, *, finish: bool = True) -> None:
        self.finish = finish
        self.calls = 0

    async def complete(self, *, messages, tools):  # type: ignore[no-untyped-def]
        del messages, tools
        self.calls += 1
        if not self.finish:
            return ModelResponse(content="silent")
        return ModelResponse(
            content=None,
            tool_calls=(
                FunctionCall(
                    "finish",
                    "finish_drift",
                    {
                        "skill_used": "research",
                        "one_line": "完成后台检查",
                        "next": "等待",
                        "message_result": "silent",
                    },
                ),
            )
        )


@pytest.mark.asyncio
async def test_drift_runs_without_agent_runtime_and_finishes() -> None:
    provider = _Provider()
    pipeline = DriftTurnPipeline(
        _Repository(),
        provider,
        tool_executor=ToolExecutor(ToolRegistry()),
        drift_selector=_Selector(),
    )  # type: ignore[arg-type]

    result = await pipeline.execute_task(
        task_id="drift-1",
        session_key="feishu:chat-1",
        payload={"chat_id": "chat-1", "activity_version": 4},
        now=NOW,
    )

    assert result.outcome == "succeeded"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_drift_without_finish_is_failed() -> None:
    pipeline = DriftTurnPipeline(
        _Repository(),
        _Provider(finish=False),
        tool_executor=ToolExecutor(ToolRegistry()),
        drift_selector=_Selector(),
    )  # type: ignore[arg-type]

    result = await pipeline.execute_task(
        task_id="drift-1",
        session_key="feishu:chat-1",
        payload={"chat_id": "chat-1", "activity_version": 4},
        now=NOW,
    )

    assert result.outcome == "failed"


@pytest.mark.asyncio
async def test_drift_message_push_restricts_next_tool_schema(tmp_path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.schemas: list[set[str]] = []

        async def complete(self, *, messages, tools):  # type: ignore[no-untyped-def]
            del messages
            names = {str(item["function"]["name"]) for item in tools}
            self.schemas.append(names)
            if len(self.schemas) == 1:
                return ModelResponse(
                    content=None,
                    tool_calls=(
                        FunctionCall("push", "message_push", {"message": "想到一件事"}),
                    )
                )
            return ModelResponse(
                content=None,
                tool_calls=(
                    FunctionCall(
                        "finish",
                        "finish_drift",
                        {
                            "skill_used": "research",
                            "one_line": "完成后台检查",
                            "next": "等待",
                            "message_result": "sent",
                        },
                    ),
                )
            )

    class Outbound:
        async def dispatch(self, _value):  # type: ignore[no-untyped-def]
            return True

    async def noop(**_kwargs):  # type: ignore[no-untyped-def]
        return "ok"

    shared = ToolRegistry(
        (
            Tool("shell", "shell", {"type": "object"}, noop),
            Tool("web_search", "web", {"type": "object"}, noop),
            Tool("recall_memory", "memory", {"type": "object"}, noop),
        )
    )
    provider = Provider()
    pipeline = DriftTurnPipeline(
        _Repository(),
        provider,
        tool_executor=ToolExecutor(shared),
        outbound=Outbound(),
        drift_selector=_Selector(),
        drift_workspace=tmp_path,
        shared_tools=shared,
    )  # type: ignore[arg-type]

    result = await pipeline.execute_task(
        task_id="drift-1",
        session_key="feishu:chat-1",
        payload={"channel": "feishu", "chat_id": "chat-1", "activity_version": 4},
        now=NOW,
    )

    assert result.outcome == "succeeded"
    assert provider.schemas[1] == {"write_file", "edit_file", "finish_drift"}
    assert not provider.schemas[1].intersection(
        {"shell", "web_search", "recall_memory", "message_push"}
    )
