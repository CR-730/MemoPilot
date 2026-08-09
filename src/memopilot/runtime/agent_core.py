"""被动 Turn 的生产入口。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from memopilot.bus.events import InboundMessage
from memopilot.runtime.engine import TurnInput, TurnResult
from memopilot.tasks.agent_task import AgentTask


class PassiveExecutor(Protocol):
    async def execute_direct(self, turn: TurnInput) -> TurnResult: ...

    async def run(
        self,
        message: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> Sequence[AgentTask]: ...


class AgentCore:
    """统一承接 Redis 被动任务与不出站的直接 Agent 请求。"""

    def __init__(self, pipeline: PassiveExecutor) -> None:
        self._pipeline = pipeline

    async def process(
        self,
        message: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> Sequence[AgentTask]:
        return await self._pipeline.run(
            message,
            key,
            dispatch_outbound=dispatch_outbound,
        )

    async def run_direct(
        self,
        *,
        session_key: str,
        content: str,
        now: datetime,
        source_ref: str,
    ) -> TurnResult:
        return await self._pipeline.execute_direct(
            TurnInput(
                session_key,
                content,
                prompt_scope="scheduled",
                received_at=now,
                allowed_tool_risks=frozenset({"read-only", "write"}),
                memory_source_ref=source_ref,
                disabled_tools=frozenset({"message_push"}),
                omit_user_turn=True,
                skip_post_memory=True,
                suppress_stream_events=True,
            )
        )


__all__ = ["AgentCore"]
