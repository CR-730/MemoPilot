from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memopilot import cli
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


async def test_run_all_starts_services_in_one_async_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    ready = asyncio.Event()

    class FakeAppService:
        async def start(self) -> None:
            events.append("app.start")
            ready.set()

    class FakeLoopService:
        def __init__(self, name: str) -> None:
            self.name = name

        async def run_forever(self) -> None:
            events.append(f"{self.name}.run")
            await ready.wait()

    class FakeAppBundle:
        service = FakeAppService()

        async def close(self) -> None:
            events.append("app.close")

    class FakeSchedulerBundle:
        service = FakeLoopService("scheduler")

        async def close(self) -> None:
            events.append("scheduler.close")

    class FakeWorkerBundle:
        service = FakeLoopService("worker")
        mcp_diagnostics: tuple[str, ...] = ()

        async def start_extensions(self) -> None:
            events.append("worker.extensions")

        async def close(self) -> None:
            events.append("worker.close")

    monkeypatch.setattr(cli, "build_app", lambda settings: FakeAppBundle())
    monkeypatch.setattr(cli, "build_scheduler", lambda settings: FakeSchedulerBundle())

    async def build_worker(_settings: object) -> FakeWorkerBundle:
        return FakeWorkerBundle()

    monkeypatch.setattr(cli, "build_worker", build_worker)

    await cli.run_all(SimpleNamespace(feishu_allow_from=()))

    assert events == [
        "worker.extensions",
        "app.start",
        "scheduler.run",
        "worker.run",
        "worker.close",
        "scheduler.close",
        "app.close",
    ]
