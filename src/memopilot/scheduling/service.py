"""定时任务用例服务。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from memopilot.runtime.outbound import DeliveryError, OutboundDispatch, OutboundPort
from memopilot.scheduling.contracts import (
    CreateSchedule,
    DueScanResult,
    ExecutionMode,
    ScheduledTask,
    ScheduleKind,
)
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.scheduling.time_rules import compute_fire_at
from memopilot.tasks.agent_task import AgentTask

if TYPE_CHECKING:
    from memopilot.runtime.engine import AgentRuntime


class SchedulerService:
    def __init__(
        self,
        repository: ScheduleRepository,
        *,
        runtime: AgentRuntime | None = None,
        outbound: OutboundPort | None = None,
    ) -> None:
        self.repository = repository
        self._runtime = runtime
        self._outbound = outbound

    def bind_executor(self, runtime: AgentRuntime, outbound: OutboundPort) -> None:
        self._runtime = runtime
        self._outbound = outbound

    def schedule(
        self,
        *,
        session_key: str,
        received_at: datetime,
        schedule_kind: str,
        when: str,
        execution_mode: str,
        message: str | None = None,
        prompt: str | None = None,
        name: str | None = None,
        timezone: str = "Asia/Shanghai",
        now: datetime | None = None,
    ) -> ScheduledTask:
        if schedule_kind not in {"at", "after", "every"}:
            raise ValueError("schedule_kind 必须为 at、after 或 every")
        if execution_mode not in {"instant", "agent"}:
            raise ValueError("execution_mode 必须为 instant 或 agent")
        if execution_mode == "instant" and not (message or "").strip():
            raise ValueError("instant 模式必须提供 message")
        if execution_mode == "agent" and not (prompt or "").strip():
            raise ValueError("agent 模式必须提供 prompt")
        received_at = (
            received_at.replace(tzinfo=ZoneInfo(timezone))
            if received_at.tzinfo is None
            else received_at
        )
        current = now or datetime.now(UTC)
        next_run_at = compute_fire_at(
            schedule_kind,
            when,
            timezone_name=timezone,
            received_at=received_at,
            now_fn=lambda: current,
        )
        payload: Mapping[str, object] = (
            {"message": message} if execution_mode == "instant" else {"prompt": prompt}
        )
        return self.repository.create(
            CreateSchedule(
                session_key=session_key,
                schedule_kind=cast(ScheduleKind, schedule_kind),
                schedule_expression=when,
                execution_mode=cast(ExecutionMode, execution_mode),
                payload=payload,
                next_run_at=next_run_at,
                timezone=timezone,
                name=name,
                created_at=received_at,
            )
        )

    def list(self, session_key: str) -> tuple[ScheduledTask, ...]:
        return self.repository.list_for_session(session_key)

    def cancel(
        self,
        session_key: str,
        *,
        task_id: str | None = None,
        name: str | None = None,
    ) -> tuple[str, ...]:
        return self.repository.cancel(session_key, task_id=task_id, name=name)

    def scan_due(self, *, now: datetime) -> DueScanResult:
        return self.repository.enqueue_due(now=now)

    async def execute_task(self, task: AgentTask, *, now: datetime) -> tuple[AgentTask, ...]:
        if self._runtime is None or self._outbound is None:
            raise RuntimeError("SchedulerService 未配置执行依赖")
        payload = task.payload
        execution_id = _required_text(payload, "execution_id")
        current = self.repository.transition_execution(execution_id, outcome="running", now=now)
        if current in {"succeeded", "failed", "cancelled"}:
            return ()
        try:
            task_payload = payload.get("payload")
            if not isinstance(task_payload, Mapping):
                raise ValueError("schedule.run payload.payload 必须是对象")
            mode = _required_text(payload, "execution_mode")
            if mode == "instant":
                text = _required_text(task_payload, "message")
            elif mode == "agent":
                from memopilot.runtime.engine import TurnInput

                result = await self._runtime.run(
                    TurnInput(
                        task.session_key,
                        _required_text(task_payload, "prompt"),
                        prompt_scope="scheduled",
                        received_at=now,
                        allowed_tool_risks=frozenset({"read-only", "write"}),
                        memory_source_ref=f"task:{task.task_id}",
                    )
                )
                if result.react.infrastructure_error:
                    self.repository.transition_execution(execution_id, outcome="failed", now=now)
                    return ()
                text = result.reply
            else:
                raise ValueError(f"未知 schedule execution_mode: {mode}")
            if not await self._outbound.dispatch(
                OutboundDispatch(
                    _required_text(payload, "channel"),
                    _required_text(payload, "chat_id"),
                    text,
                    metadata={
                        "provider_uuid": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"memopilot:schedule:{execution_id}",
                            )
                        )
                    },
                )
            ):
                raise DeliveryError("定时任务结果未明确发送成功")
            self.repository.transition_execution(execution_id, outcome="succeeded", now=now)
            return ()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, DeliveryError):
            raise
        except BaseException:
            self.repository.transition_execution(execution_id, outcome="failed", now=now)
            raise


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Agent 任务缺少 {key}")
    return value


__all__ = ["SchedulerService"]
