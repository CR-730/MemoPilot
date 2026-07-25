from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from memopilot import cli


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
