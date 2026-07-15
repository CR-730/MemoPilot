"""MCP Wake Source 的 fetch/cursor 与幂等 ACK 逻辑合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.extensions.mcp import McpCallResult, McpInvocationError


class McpCaller(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult: ...


@dataclass(frozen=True, slots=True)
class WakeSourceConfig:
    source_id: str
    server_id: str
    fetch_tool: str
    ack_tool: str
    ack_idempotent: bool
    max_batch: int = 50

    def __post_init__(self) -> None:
        if not all((self.source_id, self.server_id, self.fetch_tool, self.ack_tool)):
            raise ValueError("Wake Source 标识和工具映射不能为空")
        if not self.ack_idempotent:
            raise ValueError("Wake Source 必须声明 ACK 幂等保证")
        if self.max_batch <= 0:
            raise ValueError("Wake Source max_batch 必须大于 0")


@dataclass(frozen=True, slots=True)
class WakeEvent:
    source_id: str
    event_id: str
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WakeFetchResult:
    events: tuple[WakeEvent, ...]
    cursor: str | None


class McpWakeSource:
    def __init__(self, config: WakeSourceConfig, caller: McpCaller) -> None:
        self.config = config
        self._caller = caller

    async def fetch(self, *, cursor: str | None) -> WakeFetchResult:
        result = await self._caller.call_tool(
            self.config.fetch_tool,
            {"cursor": cursor, "limit": self.config.max_batch},
        )
        if not result.ok or result.structured is None:
            raise McpInvocationError(
                "wake_source_fetch_error",
                result.error_message or "Wake Source fetch 缺少结构化结果",
            )
        raw_events = result.structured.get("events")
        if not isinstance(raw_events, list):
            raise McpInvocationError("wake_source_contract_error", "events 必须是数组")
        events: list[WakeEvent] = []
        seen: set[str] = set()
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise McpInvocationError("wake_source_contract_error", "event 必须是对象")
            event_id = str(raw.get("event_id") or "")
            kind = str(raw.get("kind") or "")
            payload = raw.get("payload")
            if not event_id or event_id in seen or kind not in {"alert", "context", "content"}:
                raise McpInvocationError("wake_source_contract_error", "event 标识或 kind 无效")
            if not isinstance(payload, dict):
                raise McpInvocationError("wake_source_contract_error", "event payload 必须是对象")
            seen.add(event_id)
            events.append(WakeEvent(self.config.source_id, event_id, kind, dict(payload)))
        raw_cursor = result.structured.get("cursor")
        cursor_value = None if raw_cursor is None else str(raw_cursor)
        return WakeFetchResult(tuple(events), cursor_value)

    async def ack(self, event_id: str, *, operation_id: str) -> None:
        result = await self._caller.call_tool(
            self.config.ack_tool,
            {"event_id": event_id, "operation_id": operation_id},
        )
        acknowledged = result.structured is not None and result.structured.get("ok") is True
        if not result.ok or not acknowledged:
            raise McpInvocationError(
                "wake_source_ack_error",
                result.error_message or "Wake Source ACK 未确认成功",
            )


def stable_ack_operation_id(source_id: str, event_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"memopilot:wake-ack:{source_id}:{event_id}"))


__all__ = [
    "McpCaller",
    "McpWakeSource",
    "WakeEvent",
    "WakeFetchResult",
    "WakeSourceConfig",
    "stable_ack_operation_id",
]
