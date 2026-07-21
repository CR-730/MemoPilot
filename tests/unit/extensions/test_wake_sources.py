from __future__ import annotations

import pytest

from memopilot.extensions.mcp import McpCallResult
from memopilot.extensions.wake_sources import (
    McpWakeSource,
    WakeSourceConfig,
    stable_ack_operation_id,
)


class _Caller:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        if name == "fetch_events":
            return McpCallResult(
                structured={
                    "events": [
                        {
                            "event_id": "e1",
                            "kind": "content",
                            "occurred_at": "2026-07-16T12:00:00Z",
                            "payload": {"title": "A"},
                        }
                    ],
                    "next_cursor": "next",
                }
            )
        return McpCallResult(structured={"acknowledged": True})


def test_wake_source_requires_idempotent_ack_contract() -> None:
    with pytest.raises(ValueError, match="ACK 幂等"):
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch_events",
            ack_tool="ack_event",
            capability="wake_source.v1",
            ack_idempotent=False,
        )


async def test_fetch_and_ack_use_cursor_batch_and_stable_operation_id() -> None:
    caller = _Caller()
    source = McpWakeSource(
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch_events",
            ack_tool="ack_event",
            capability="wake_source.v1",
            ack_idempotent=True,
            max_batch=20,
        ),
        caller,
    )

    fetched = await source.fetch(cursor="old")
    operation_id = stable_ack_operation_id("news", "e1")
    await source.ack("e1", operation_id=operation_id)

    assert fetched.cursor == "next"
    assert [
        (event.event_id, event.kind, event.occurred_at) for event in fetched.events
    ] == [("e1", "content", "2026-07-16T12:00:00Z")]
    assert caller.calls == [
        ("fetch_events", {"cursor": "old", "limit": 20}),
        ("ack_event", {"event_id": "e1", "operation_id": operation_id}),
    ]
    assert operation_id == stable_ack_operation_id("news", "e1")


async def test_fetch_rejects_missing_or_invalid_occurred_at() -> None:
    caller = _Caller()
    source = McpWakeSource(
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch_events",
            ack_tool="ack_event",
            capability="wake_source.v1",
            ack_idempotent=True,
        ),
        caller,
    )
    original = caller.call_tool

    async def invalid(name: str, arguments: dict[str, object]) -> McpCallResult:
        result = await original(name, arguments)
        assert result.structured is not None
        result.structured["events"][0]["occurred_at"] = "not-a-time"
        return result

    caller.call_tool = invalid  # type: ignore[method-assign]
    with pytest.raises(Exception, match="occurred_at"):
        await source.fetch(cursor=None)


@pytest.mark.parametrize("next_cursor", [{"page": 2}, ["next"], 2, "", "   "])
async def test_fetch_rejects_non_string_or_blank_next_cursor(next_cursor: object) -> None:
    caller = _Caller()
    source = McpWakeSource(
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch_events",
            ack_tool="ack_event",
            capability="wake_source.v1",
            ack_idempotent=True,
        ),
        caller,
    )
    original = caller.call_tool

    async def invalid(name: str, arguments: dict[str, object]) -> McpCallResult:
        result = await original(name, arguments)
        assert result.structured is not None
        result.structured["next_cursor"] = next_cursor
        return result

    caller.call_tool = invalid  # type: ignore[method-assign]
    with pytest.raises(Exception) as exc_info:
        await source.fetch(cursor=None)
    assert getattr(exc_info.value, "error_type", None) == "wake_source_contract_error"


def test_wake_source_requires_capability_and_positive_timeouts() -> None:
    with pytest.raises(ValueError, match="capability"):
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch",
            ack_tool="ack",
            capability="",
            ack_idempotent=True,
        )
    with pytest.raises(ValueError, match="超时"):
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch",
            ack_tool="ack",
            capability="wake_source.v1",
            ack_idempotent=True,
            fetch_timeout_seconds=0,
        )

    with pytest.raises(ValueError, match="wake_source.v1"):
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch",
            ack_tool="ack",
            capability="wake_source.v2",
            ack_idempotent=True,
        )


def test_wake_source_revalidates_capability_when_registered() -> None:
    config = WakeSourceConfig(
        source_id="news",
        server_id="fake",
        fetch_tool="fetch",
        ack_tool="ack",
        capability="wake_source.v1",
        ack_idempotent=True,
    )
    object.__setattr__(config, "capability", "wake_source.v2")

    with pytest.raises(ValueError, match="wake_source.v1"):
        McpWakeSource(config, _Caller())
