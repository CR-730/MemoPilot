"""独立的 Drift Scan -> Prepare -> Execute -> Finish 管线。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.persistence.conversation import ConversationRepository, StaleActivityError
from memopilot.proactive.drift import DRIFT_SYSTEM_PROMPT
from memopilot.proactive.drift_tools import DriftRunState, build_drift_tool_registry
from memopilot.proactive.store import ProactiveRepository
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.outbound import OutboundDispatch, OutboundPort
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.react import AfterStepControl, BeforeStepControl, ReActEngine, ToolCallRecord
from memopilot.runtime.tools import ToolExecutor, ToolRegistry


@dataclass(frozen=True, slots=True)
class DriftResult:
    outcome: str


class DriftSkillSelector(Protocol):
    async def select(self) -> str: ...


class _DriftObserver:
    def __init__(self, state: DriftRunState) -> None:
        self._state = state
        self.visible: frozenset[str] | None = None

    def set_visible_tool_names(self, names: frozenset[str]) -> None:
        self.visible = names

    async def before_step(self, iteration: int, messages: object) -> BeforeStepControl:
        del iteration, messages
        return BeforeStepControl()

    async def after_step(
        self, iteration: int, response: object, records: Sequence[ToolCallRecord], messages: object
    ) -> AfterStepControl:
        del iteration, response, messages
        successful = {record.call.name for record in records if record.observation.ok}
        if "message_push" in successful:
            self.visible = frozenset({"write_file", "edit_file", "finish_drift"})
        return AfterStepControl(
            "finish_drift" in successful,
            "finish_drift",
            self.visible if "message_push" in successful else None,
        )


class DriftTurnPipeline:
    def __init__(
        self,
        repository: ConversationRepository,
        provider: ChatProvider,
        *,
        tool_executor: ToolExecutor,
        outbound: OutboundPort | None = None,
        drift_selector: DriftSkillSelector | None = None,
        drift_workspace: Path | None = None,
        drift_builtin_skills: Path | None = None,
        drift_repository: ProactiveRepository | None = None,
        shared_tools: ToolRegistry | None = None,
        connected_mcp_servers: Callable[[], frozenset[str]] | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.repository, self.provider, self.tool_executor = repository, provider, tool_executor
        self.outbound, self.drift_selector = outbound, drift_selector
        self.drift_workspace, self.drift_builtin_skills = drift_workspace, drift_builtin_skills
        self.drift_repository, self.shared_tools = drift_repository, shared_tools
        self.connected_mcp_servers, self.max_iterations = connected_mcp_servers, max_iterations

    async def execute_task(
        self, *, task_id: str, session_key: str, payload: dict[str, object], now: datetime
    ) -> DriftResult:
        if self.drift_selector is None:
            raise RuntimeError("未配置 Drift Skill Selector")
        activity_version = _required_integer(payload, "activity_version")

        def assert_current() -> None:
            if self.repository.get_activity_version(session_key) != activity_version:
                raise StaleActivityError("用户活跃状态已经变化，丢弃陈旧 Drift 任务")

        assert_current()
        skill_name = await self.drift_selector.select()  # Scan
        assert_current()
        state = DriftRunState(frozenset({skill_name}))

        async def send_message(text: str, media: list[str]) -> bool:
            if self.outbound is None:
                return False
            assert_current()
            sent = await self.outbound.dispatch(
                OutboundDispatch(
                    _required_text(payload, "channel"),
                    _required_text(payload, "chat_id"),
                    text,
                    media=media,
                    metadata={
                        "provider_uuid": str(uuid5(NAMESPACE_URL, f"memopilot:drift:{task_id}"))
                    },
                )
            )
            if sent:
                self.repository.commit_direct_assistant(
                    session_key=session_key,
                    channel=_required_text(payload, "channel"),
                    chat_id=_required_text(payload, "chat_id"),
                    content=text,
                    media=tuple(media),
                    timestamp=now,
                    source_ref=f"drift:{task_id}",
                    metadata={
                        "proactive": True,
                        "tools_used": ["message_push"],
                        "evidence_item_ids": [],
                        "state_summary_tag": "none",
                    },
                )
            return sent

        tools = build_drift_tool_registry(  # Prepare
            workspace=_resolve_drift_workspace(payload.get("workspace"), self.drift_workspace),
            builtin_skills_dir=self.drift_builtin_skills,
            state=state,
            send_message=send_message,
            shared_tools=self.shared_tools,
            connected_servers=self.connected_mcp_servers()
            if self.connected_mcp_servers
            else frozenset(),
        )
        observer = _DriftObserver(state)
        try:
            result = await ReActEngine(  # Execute
                self.provider,
                tools,
                tool_executor=self.tool_executor.for_registry(tools),
                max_iterations=self.max_iterations,
                observer=observer,
                session_key=session_key,
                source="drift",
                request_text=f"${skill_name}",
                assert_current=assert_current,
            ).run((ChatMessage.system(DRIFT_SYSTEM_PROMPT), ChatMessage.user(f"${skill_name}")))
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
        outcome = "succeeded" if state.finished and not result.infrastructure_error else "failed"
        self._complete(
            session_key, task_id, skill_name, outcome, state.finish_payload or {}, now
        )  # Finish
        return DriftResult(outcome)

    def _complete(
        self,
        session_key: str,
        task_id: str,
        skill_name: str,
        outcome: str,
        result: dict[str, str],
        now: datetime,
    ) -> None:
        if self.drift_repository is not None:
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
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        return int(value)
    raise ValueError(f"后台任务缺少 {key}")


def _resolve_drift_workspace(value: object, configured: Path | None) -> Path:
    return Path(str(value or configured or ".")).resolve()


__all__ = ["DriftTurnPipeline", "DriftResult", "DriftSkillSelector"]
