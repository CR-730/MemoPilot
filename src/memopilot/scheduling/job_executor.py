"""Proactive、Schedule 与 Drift 系统 Job 的执行路由。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from memopilot.delivery.feishu import DeliveryOutcome
from memopilot.proactive.drift import DRIFT_SYSTEM_PROMPT
from memopilot.proactive.drift_runtime import DriftRunState, build_drift_tool_registry
from memopilot.proactive.store import ProactiveRepository
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.persistence import OperationalStepSink
from memopilot.runtime.tools import ToolRegistry
from memopilot.runtime.worker import SystemJobResult
from memopilot.tasks.operational import (
    FenceToken,
    LostLeaseError,
    OperationalRepository,
    RunClaim,
)


class ResponseDispatcher(Protocol):
    async def dispatch(
        self,
        *,
        claim: RunClaim,
        lease: FenceToken,
        text: str,
    ) -> Any: ...


class ProactiveJobHandler(Protocol):
    async def execute(
        self,
        *,
        payload: dict[str, object],
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
    ) -> str: ...


class DriftSkillSelector(Protocol):
    async def select(self) -> str: ...


class SystemJobRouter:
    def __init__(
        self,
        repository: OperationalRepository,
        runtime: AgentRuntime,
        *,
        dispatcher: ResponseDispatcher | None = None,
        proactive_handler: ProactiveJobHandler | None = None,
        drift_selector: DriftSkillSelector | None = None,
        drift_workspace: Path | None = None,
        drift_builtin_skills: Path | None = None,
        drift_repository: ProactiveRepository | None = None,
        shared_tools: ToolRegistry | None = None,
        connected_mcp_servers: Callable[[], frozenset[str]] | None = None,
    ) -> None:
        self.repository = repository
        self.runtime = runtime
        self.dispatcher = dispatcher
        self.proactive_handler = proactive_handler
        self.drift_selector = drift_selector
        self.drift_workspace = drift_workspace
        self.drift_builtin_skills = drift_builtin_skills
        self.drift_repository = drift_repository
        self.shared_tools = shared_tools
        self.connected_mcp_servers = connected_mcp_servers

    async def execute(
        self,
        *,
        kind: str,
        payload: dict[str, object],
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
    ) -> SystemJobResult:
        self.repository.assert_current_fence(lease)
        if kind == "schedule.run":
            return await self._run_schedule(payload, claim, lease, turn, now)
        if kind == "drift.run":
            return await self._run_drift(
                payload, claim, lease, turn, now, ensure_audit=True
            )
        if kind == "proactive.tick":
            if self.proactive_handler is None:
                raise RuntimeError("未配置 Proactive Job Handler")
            outcome = await self.proactive_handler.execute(
                payload=payload,
                claim=claim,
                lease=lease,
                turn=turn,
                now=now,
            )
            if outcome == "drift":
                return await self._run_drift(
                    payload, claim, lease, turn, now, ensure_audit=False
                )
            return SystemJobResult(outcome=outcome)
        raise ValueError(f"不支持的系统任务: {kind}")

    async def _run_schedule(
        self,
        payload: Mapping[str, object],
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
    ) -> SystemJobResult:
        execution_id = _required_text(payload, "execution_id")
        current = self.repository.transition_scheduled_execution(
            execution_id,
            job_id=claim.job_id,
            lease=lease,
            outcome="running",
            now=now,
        )
        if current in {"succeeded", "failed", "cancelled", "needs_review"}:
            return SystemJobResult(outcome=current)
        try:
            return await self._execute_running_schedule(
                execution_id,
                payload,
                claim,
                lease,
                turn,
                now,
            )
        except (asyncio.CancelledError, LostLeaseError):
            raise
        except Exception:
            self.repository.transition_scheduled_execution(
                execution_id,
                job_id=claim.job_id,
                lease=lease,
                outcome="failed",
                now=now,
            )
            raise

    async def _execute_running_schedule(
        self,
        execution_id: str,
        payload: Mapping[str, object],
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
    ) -> SystemJobResult:
        mode = _required_text(payload, "execution_mode")
        task_payload = payload.get("payload")
        if not isinstance(task_payload, Mapping):
            raise ValueError("schedule.run payload.payload 必须是对象")
        self.repository.assert_current_fence(lease)
        turn_result = None
        if mode == "instant":
            text = _required_text(task_payload, "message")
        elif mode == "agent":
            assert_current = self._system_checkpoint(claim, lease)
            prompt = _required_text(task_payload, "prompt")
            scheduled_turn = replace(
                turn,
                content=prompt,
                current_user_content=None,
                prompt_scope="scheduled",
                memory_source_ref=f"run:{claim.run_id}",
                allowed_tool_risks=frozenset({"read-only", "write"}),
            )
            turn_result = await self.runtime.run(
                scheduled_turn,
                step_sink=OperationalStepSink(
                    self.repository,
                    run_id=claim.run_id,
                    lease=lease,
                    clock=lambda: now,
                ),
                execution_assert_current=assert_current,
                memory_assert_current=assert_current,
                memory_fenced_write=lambda: self.repository.fenced_write(lease),
            )
            if turn_result.react.infrastructure_error:
                outcome = "failed"
                self.repository.transition_scheduled_execution(
                    execution_id,
                    job_id=claim.job_id,
                    lease=lease,
                    outcome=outcome,
                    now=now,
                )
                return SystemJobResult(outcome, turn_result)
            text = turn_result.reply
        else:
            raise ValueError(f"未知 schedule execution_mode: {mode}")
        if self.dispatcher is None:
            raise RuntimeError("未配置系统任务消息发送器")
        self.repository.assert_current_fence(lease)
        delivery = await self.dispatcher.dispatch(
            claim=claim,
            lease=lease,
            text=text,
        )
        outcome = _delivery_outcome(delivery.outcome)
        self.repository.transition_scheduled_execution(
            execution_id,
            job_id=claim.job_id,
            lease=lease,
            outcome=outcome,
            now=now,
        )
        return SystemJobResult(outcome, turn_result)

    async def _run_drift(
        self,
        payload: Mapping[str, object],
        claim: RunClaim,
        lease: FenceToken,
        turn: TurnInput,
        now: datetime,
        *,
        ensure_audit: bool,
    ) -> SystemJobResult:
        if self.drift_selector is None:
            raise RuntimeError("未配置 Drift Skill Selector")
        assert_current = self._system_checkpoint(claim, lease)
        assert_current()
        skill_name = await self.drift_selector.select()
        assert_current()
        state = DriftRunState(frozenset({skill_name}))
        if ensure_audit and self.drift_repository is not None:
            self.drift_repository.mark_drift_started(
                session_key=turn.session_key,
                job_id=claim.job_id,
                started_at=now,
            )

        async def send_message(text: str, media: list[str]) -> bool:
            del media
            if self.dispatcher is None:
                return False
            assert_current()
            delivery = await self.dispatcher.dispatch(
                claim=claim,
                lease=lease,
                text=text,
            )
            return _delivery_outcome(delivery.outcome) == "succeeded"

        workspace = _resolve_drift_workspace(
            payload.get("workspace"), self.drift_workspace
        )
        drift_tools = build_drift_tool_registry(
            workspace=workspace,
            builtin_skills_dir=self.drift_builtin_skills,
            state=state,
            send_message=send_message,
            shared_tools=self.shared_tools,
            connected_servers=(
                self.connected_mcp_servers()
                if self.connected_mcp_servers is not None
                else frozenset()
            ),
        )
        try:
            result = await self.runtime.run(
                replace(
                    turn,
                    content=f"${skill_name}",
                    system_prompt=DRIFT_SYSTEM_PROMPT,
                    current_user_content=None,
                    prompt_scope="background",
                    memory_source_ref=f"run:{claim.run_id}",
                    allowed_tool_risks=frozenset({"read-only", "write"}),
                ),
                tools=drift_tools,
                step_sink=OperationalStepSink(
                    self.repository,
                    run_id=claim.run_id,
                    lease=lease,
                    clock=lambda: now,
                ),
                execution_assert_current=assert_current,
                memory_assert_current=assert_current,
                memory_fenced_write=lambda: self.repository.fenced_write(lease),
            )
        except BaseException as exc:
            with suppress(Exception):
                self._complete_drift(
                    turn.session_key,
                    claim.job_id,
                    skill_name,
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    state.finish_payload or {},
                    now,
                )
            raise
        outcome = (
            "failed"
            if result.react.infrastructure_error or not state.finished
            else "succeeded"
        )
        self._complete_drift(
            turn.session_key,
            claim.job_id,
            skill_name,
            outcome,
            state.finish_payload or {},
            now,
        )
        return SystemJobResult(outcome, result)

    def _complete_drift(
        self,
        session_key: str,
        job_id: str,
        skill_name: str,
        outcome: str,
        result: dict[str, str],
        now: datetime,
    ) -> None:
        if self.drift_repository is None:
            return
        self.drift_repository.complete_drift(
            session_key=session_key,
            job_id=job_id,
            skill_name=skill_name,
            outcome=outcome,
            result=result,
            completed_at=now,
        )

    def _system_checkpoint(
        self,
        claim: RunClaim,
        lease: FenceToken,
    ) -> Callable[[], None]:
        job = self.repository.get_job(claim.job_id)
        if job is None:
            raise KeyError(claim.job_id)

        def assert_current() -> None:
            self.repository.assert_current_fence_and_activity(
                lease,
                expected_activity_version=job.activity_version,
            )

        return assert_current


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"系统任务缺少 {key}")
    return value


def _resolve_drift_workspace(value: object, configured: Path | None) -> Path:
    return Path(str(value or configured or ".")).resolve()


def _delivery_outcome(outcome: object) -> str:
    normalized = cast(str, getattr(outcome, "value", outcome))
    return {
        DeliveryOutcome.CONFIRMED.value: "succeeded",
        DeliveryOutcome.CANCELLED.value: "cancelled",
        DeliveryOutcome.FAILED.value: "failed",
        DeliveryOutcome.NEEDS_REVIEW.value: "needs_review",
    }[normalized]


__all__ = ["SystemJobRouter", "ProactiveJobHandler"]
