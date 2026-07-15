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
                        {"event_id": "e1", "kind": "content", "payload": {"title": "A"}}
                    ],
                    "cursor": "next",
                }
            )
        return McpCallResult(structured={"ok": True})


def test_wake_source_requires_idempotent_ack_contract() -> None:
    with pytest.raises(ValueError, match="ACK 幂等"):
        WakeSourceConfig(
            source_id="news",
            server_id="fake",
            fetch_tool="fetch_events",
            ack_tool="ack_event",
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
            ack_idempotent=True,
            max_batch=20,
        ),
        caller,
    )

    fetched = await source.fetch(cursor="old")
    operation_id = stable_ack_operation_id("news", "e1")
    await source.ack("e1", operation_id=operation_id)

    assert fetched.cursor == "next"
    assert [(event.event_id, event.kind) for event in fetched.events] == [("e1", "content")]
    assert caller.calls == [
        ("fetch_events", {"cursor": "old", "limit": 20}),
        ("ack_event", {"event_id": "e1", "operation_id": operation_id}),
    ]
    assert operation_id == stable_ack_operation_id("news", "e1")

