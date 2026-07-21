"""MCP Wake Source 的 fetch/cursor 与幂等 ACK 逻辑合同。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from memopilot.extensions.mcp import McpCallResult, McpInvocationError

_SUPPORTED_CAPABILITIES = frozenset({"wake_source.v1"})


class McpCaller(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult: ...


@dataclass(frozen=True, slots=True)
class WakeSourceConfig:
    source_id: str
    server_id: str
    fetch_tool: str
    ack_tool: str
    capability: str
    ack_idempotent: bool
    max_batch: int = 50
    fetch_timeout_seconds: float = 30.0
    ack_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not all((self.source_id, self.server_id, self.fetch_tool, self.ack_tool)):
            raise ValueError("Wake Source 标识和工具映射不能为空")
        _validate_capability(self.capability)
        if not self.ack_idempotent:
            raise ValueError("Wake Source 必须声明 ACK 幂等保证")
        if self.max_batch <= 0:
            raise ValueError("Wake Source max_batch 必须大于 0")
        if self.fetch_timeout_seconds <= 0 or self.ack_timeout_seconds <= 0:
            raise ValueError("Wake Source fetch/ACK 超时必须大于 0")


@dataclass(frozen=True, slots=True)
class WakeEvent:
    source_id: str
    event_id: str
    kind: str
    occurred_at: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WakeFetchResult:
    events: tuple[WakeEvent, ...]
    cursor: str | None


class McpWakeSource:
    def __init__(self, config: WakeSourceConfig, caller: McpCaller) -> None:
        _validate_capability(config.capability)
        self.config = config
        self._caller = caller

    async def fetch(self, *, cursor: str | None) -> WakeFetchResult:
        try:
            async with asyncio.timeout(self.config.fetch_timeout_seconds):
                result = await self._caller.call_tool(
                    self.config.fetch_tool,
                    {"cursor": cursor, "limit": self.config.max_batch},
                )
        except TimeoutError as exc:
            raise McpInvocationError(
                "wake_source_fetch_timeout", "Wake Source fetch 超时"
            ) from exc
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
            occurred_at = str(raw.get("occurred_at") or "")
            payload = raw.get("payload")
            if not event_id or event_id in seen or kind not in {"alert", "context", "content"}:
                raise McpInvocationError("wake_source_contract_error", "event 标识或 kind 无效")
            if not isinstance(payload, dict):
                raise McpInvocationError("wake_source_contract_error", "event payload 必须是对象")
            if not _valid_occurred_at(occurred_at):
                raise McpInvocationError(
                    "wake_source_contract_error",
                    "event occurred_at 必须是带时区的 ISO 8601 时间",
                )
            seen.add(event_id)
            events.append(
                WakeEvent(
                    self.config.source_id,
                    event_id,
                    kind,
                    occurred_at,
                    dict(payload),
                )
            )
        raw_cursor = result.structured.get("next_cursor")
        if raw_cursor is not None and (
            not isinstance(raw_cursor, str) or not raw_cursor.strip()
        ):
            raise McpInvocationError(
                "wake_source_contract_error",
                "next_cursor 必须是 null 或非空字符串",
            )
        cursor_value = raw_cursor
        return WakeFetchResult(tuple(events), cursor_value)

    async def ack(self, event_id: str, *, operation_id: str) -> None:
        try:
            async with asyncio.timeout(self.config.ack_timeout_seconds):
                result = await self._caller.call_tool(
                    self.config.ack_tool,
                    {"event_id": event_id, "operation_id": operation_id},
                )
        except TimeoutError as exc:
            raise McpInvocationError(
                "wake_source_ack_timeout", "Wake Source ACK 超时"
            ) from exc
        acknowledged = (
            result.structured is not None
            and result.structured.get("acknowledged") is True
        )
        if not result.ok or not acknowledged:
            raise McpInvocationError(
                "wake_source_ack_error",
                result.error_message or "Wake Source ACK 未确认成功",
            )


def stable_ack_operation_id(source_id: str, event_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"memopilot:wake-ack:{source_id}:{event_id}"))


def _valid_occurred_at(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_capability(capability: str) -> None:
    if capability not in _SUPPORTED_CAPABILITIES:
        raise ValueError("Wake Source capability 仅支持 wake_source.v1")


__all__ = [
    "McpCaller",
    "McpWakeSource",
    "WakeEvent",
    "WakeFetchResult",
    "WakeSourceConfig",
    "stable_ack_operation_id",
]
