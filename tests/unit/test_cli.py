from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from memopilot.cli import execute_effect_action


async def test_effect_cli_keeps_confirm_and_retry_as_distinct_actions() -> None:
    service = AsyncMock()
    service.confirm.return_value = SimpleNamespace(operation_id="op-1", state="confirmed")
    service.retry.return_value = SimpleNamespace(
        outcome="confirmed",
        effect=SimpleNamespace(operation_id="op-1", state="confirmed"),
    )

    confirmed = await execute_effect_action(
        service,
        action="confirm",
        operation_id="op-1",
        message_id="om-observed",
    )
    retried = await execute_effect_action(
        service,
        action="retry",
        operation_id="op-1",
    )

    service.confirm.assert_awaited_once_with("op-1", message_id="om-observed")
    service.retry.assert_awaited_once_with("op-1")
    assert confirmed["state"] == "confirmed"
    assert retried["outcome"] == "confirmed"
