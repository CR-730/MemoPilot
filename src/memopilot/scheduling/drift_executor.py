"""Drift 后台任务执行器。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

from memopilot.proactive.drift import DRIFT_SYSTEM_PROMPT
from memopilot.proactive.drift_runtime import DriftRunState, build_drift_tool_registry
from memopilot.proactive.store import ProactiveRepository
from memopilot.runtime.engine import AgentRuntime, TurnInput, TurnResult
from memopilot.runtime.outbound import OutboundDispatch, OutboundPort
from memopilot.runtime.tools import ToolRegistry
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository


@dataclass(frozen=True, slots=True)
class DriftResult:
    outcome: str
    turn_result: TurnResult | None = None


class DriftSkillSelector(Protocol):
    async def select(self) -> str: ...


class DriftExecutor:
    """执行 Drift；定时、主动和记忆任务由各自领域处理器负责。"""

    def __init__(
        self,
        repository: OperationalRepository,
        runtime: AgentRuntime,
        *,
        outbound: OutboundPort | None = None,
        drift_selector: DriftSkillSelector | None = None,
        drift_workspace: Path | None = None,
        drift_builtin_skills: Path | None = None,
        drift_repository: ProactiveRepository | None = None,
        shared_tools: ToolRegistry | None = None,
        connected_mcp_servers: Callable[[], frozenset[str]] | None = None,
    ) -> None:
        self.repository = repository
        self.runtime = runtime
        self.outbound = outbound
        self.drift_selector = drift_selector
        self.drift_workspace = drift_workspace
        self.drift_builtin_skills = drift_builtin_skills
        self.drift_repository = drift_repository
        self.shared_tools = shared_tools
        self.connected_mcp_servers = connected_mcp_servers

    async def execute_task(
        self,
        *,
        task_id: str,
        session_key: str,
        payload: dict[str, object],
        lease: SessionLease,
        now: datetime,
    ) -> DriftResult:
        if self.drift_selector is None:
            raise RuntimeError("未配置 Drift Skill Selector")
        activity_version = _required_integer(payload, "activity_version")

        def assert_current() -> None:
            self.repository.assert_current_fence_and_activity(
                lease,
                expected_activity_version=activity_version,
            )

        assert_current()
        skill_name = await self.drift_selector.select()
        assert_current()
        state = DriftRunState(frozenset({skill_name}))

        async def send_message(text: str, media: list[str]) -> bool:
            if self.outbound is None:
                return False
            assert_current()
            return await self.outbound.dispatch(
                OutboundDispatch(
                    channel=_required_text(payload, "channel"),
                    chat_id=_required_text(payload, "chat_id"),
                    content=text,
                    media=media,
                )
            )

        workspace = _resolve_drift_workspace(payload.get("workspace"), self.drift_workspace)
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
        turn = TurnInput(
            session_key=session_key,
            content=f"${skill_name}",
            system_prompt=DRIFT_SYSTEM_PROMPT,
            current_user_content=None,
            prompt_scope="background",
            received_at=now,
            memory_source_ref=f"task:{task_id}",
            allowed_tool_risks=frozenset({"read-only", "write"}),
        )
        try:
            result = await self.runtime.run(
                replace(turn),
                tools=drift_tools,
                execution_assert_current=assert_current,
                memory_assert_current=assert_current,
                memory_fenced_write=lambda: self.repository.fenced_write(lease),
            )
        except BaseException as exc:
            with suppress(Exception):
                self._complete(
                    session_key,
                    task_id,
                    skill_name,
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    state.finish_payload or {},
                    now,
                )
            raise

        outcome = (
            "failed" if result.react.infrastructure_error or not state.finished else "succeeded"
        )
        self._complete(
            session_key,
            task_id,
            skill_name,
            outcome,
            state.finish_payload or {},
            now,
        )
        return DriftResult(outcome, result)

    def _complete(
        self,
        session_key: str,
        task_id: str,
        skill_name: str,
        outcome: str,
        result: dict[str, str],
        now: datetime,
    ) -> None:
        if self.drift_repository is None:
            return
        self.drift_repository.complete_drift(
            session_key=session_key,
            task_id=task_id,
            skill_name=skill_name,
            outcome=outcome,
            result=result,
            completed_at=now,
        )


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"后台任务缺少 {key}")
    return value


def _required_integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"后台任务缺少 {key}")


def _resolve_drift_workspace(value: object, configured: Path | None) -> Path:
    return Path(str(value or configured or ".")).resolve()


__all__ = ["DriftExecutor", "DriftResult", "DriftSkillSelector"]
