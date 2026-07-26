from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from memopilot.runtime.engine import LifecyclePhase, RuntimeTraceEvent
from memopilot.runtime.persistence import OperationalStepSink


class _Repository:
    def __init__(self) -> None:
        self.states: list[str] = []

    def append_step(self, _run_id: str, **kwargs: object) -> None:
        self.states.append(str(kwargs["state"]))


@pytest.mark.parametrize(
    ("runtime_state", "persisted_state"),
    [
        ("succeeded", "succeeded"),
        ("error", "failed"),
        ("denied", "skipped"),
    ],
)
async def test_step_sink_normalizes_tool_statuses_for_operational_storage(
    runtime_state: str,
    persisted_state: str,
) -> None:
    repository = _Repository()
    sink = OperationalStepSink(
        repository,  # type: ignore[arg-type]
        run_id="run-1",
        lease=SimpleNamespace(session_key="cli:1", owner_id="worker-1", epoch=1),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 7, 26, tzinfo=UTC),
    )

    await sink.record(
        RuntimeTraceEvent(
            phase=LifecyclePhase.AFTER_STEP,
            step_type="tool",
            state=runtime_state,
        )
    )

    assert repository.states == [persisted_state]
