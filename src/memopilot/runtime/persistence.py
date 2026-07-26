"""Runtime Trace 到 operational.db steps 的适配。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from memopilot.runtime.engine import RuntimeTraceEvent
from memopilot.tasks.operational import FenceToken, OperationalRepository


class OperationalStepSink:
    def __init__(
        self,
        repository: OperationalRepository,
        *,
        run_id: str,
        lease: FenceToken,
        clock: Callable[[], datetime],
    ) -> None:
        self._repository = repository
        self._run_id = run_id
        self._lease = lease
        self._clock = clock

    async def record(self, event: RuntimeTraceEvent) -> None:
        state = {"error": "failed", "denied": "skipped"}.get(event.state, event.state)
        self._repository.append_step(
            self._run_id,
            lease=self._lease,
            phase=event.phase.value,
            step_type=event.step_type,
            state=state,
            tool_name=event.tool_name,
            input=event.input,
            observation=event.observation,
            now=self._clock(),
        )


__all__ = ["OperationalStepSink"]
