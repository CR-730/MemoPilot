from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.engine import TurnResult
from memopilot.runtime.react import ReActResult
from memopilot.scheduling.drift_executor import DriftExecutor

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _result(*, infrastructure_error: bool = False) -> TurnResult:
    react = ReActResult(
        reply="完成",
        messages=(),
        iterations=1,
        tool_chain=(),
        exit_reason="infrastructure_error" if infrastructure_error else "completed",
        infrastructure_error=infrastructure_error,
    )
    return TurnResult("完成", (), react, (), ())


class _Repository:
    def __init__(self) -> None:
        self.checks: list[int] = []

    def assert_current_fence_and_activity(
        self,
        lease: Any,
        *,
        expected_activity_version: int,
    ) -> None:
        del lease
        self.checks.append(expected_activity_version)

    def fenced_write(self, lease: Any):
        del lease
        return SimpleNamespace(__enter__=lambda: None, __exit__=lambda *args: None)


class _Runtime:
    def __init__(self, *, finish: bool = True, fail: bool = False) -> None:
        self.finish = finish
        self.fail = fail
        self.turn = None

    async def run(self, turn, **kwargs):
        self.turn = turn
        if self.fail:
            raise RuntimeError("runtime failed")
        if self.finish:
            finished = await kwargs["tools"].execute(
                FunctionCall(
                    "finish",
                    "finish_drift",
                    {
                        "skill_used": "research",
                        "one_line": "完成后台检查",
                        "next": "等待",
                        "message_result": "silent",
                    },
                )
            )
            assert finished.ok
        return _result()


class _Selector:
    async def select(self) -> str:
        return "research"


@pytest.mark.asyncio
async def test_drift_executes_direct_task_without_job_or_run() -> None:
    repository = _Repository()
    runtime = _Runtime()
    router = DriftExecutor(
        repository,  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        drift_selector=_Selector(),
    )

    result = await router.execute_task(
        task_id="drift-1",
        session_key="feishu:chat-1",
        payload={"chat_id": "chat-1", "activity_version": 4},
        lease=SimpleNamespace(session_key="feishu:chat-1", owner_id="runner", epoch=1),
        now=NOW,
    )

    assert result.outcome == "succeeded"
    assert runtime.turn.memory_source_ref == "task:drift-1"
    assert repository.checks and set(repository.checks) == {4}


@pytest.mark.asyncio
async def test_drift_without_finish_is_failed() -> None:
    router = DriftExecutor(
        _Repository(),  # type: ignore[arg-type]
        _Runtime(finish=False),  # type: ignore[arg-type]
        drift_selector=_Selector(),
    )

    result = await router.execute_task(
        task_id="drift-1",
        session_key="feishu:chat-1",
        payload={"chat_id": "chat-1", "activity_version": 1},
        lease=SimpleNamespace(session_key="feishu:chat-1", owner_id="runner", epoch=1),
        now=NOW,
    )

    assert result.outcome == "failed"


@pytest.mark.asyncio
async def test_drift_runtime_error_is_not_hidden() -> None:
    router = DriftExecutor(
        _Repository(),  # type: ignore[arg-type]
        _Runtime(fail=True),  # type: ignore[arg-type]
        drift_selector=_Selector(),
    )

    with pytest.raises(RuntimeError, match="runtime failed"):
        await router.execute_task(
            task_id="drift-1",
            session_key="feishu:chat-1",
            payload={"chat_id": "chat-1", "activity_version": 1},
            lease=SimpleNamespace(session_key="feishu:chat-1", owner_id="runner", epoch=1),
            now=NOW,
        )
