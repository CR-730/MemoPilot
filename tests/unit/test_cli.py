from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from memopilot import cli


async def test_default_run_starts_service_terminal_without_embedded_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeShutdownBridge:
        def __init__(self, _loop: object, _event: object) -> None:
            return None

        def install(self) -> None:
            return None

        def mark_shutdown_complete(self) -> None:
            return None

        def uninstall(self) -> None:
            return None

    async def fake_run_all(
        settings: object,
        *,
        interactive: bool,
        shutdown_event: asyncio.Event,
    ) -> None:
        captured["settings"] = settings
        captured["interactive"] = interactive
        captured["shutdown_event"] = shutdown_event

    settings = object()
    monkeypatch.setattr(cli, "load_settings", lambda *_args, **_kwargs: settings)
    monkeypatch.setattr(cli, "_WindowsConsoleShutdownBridge", FakeShutdownBridge)
    monkeypatch.setattr(cli, "run_all", fake_run_all)

    await cli._run(SimpleNamespace(command="run", config=None, workspace=None))

    assert captured["settings"] is settings
    assert captured["interactive"] is False
    assert isinstance(captured["shutdown_event"], asyncio.Event)


def test_configure_logging_hides_http_handshake_noise() -> None:
    cli._configure_logging()

    handler = logging.getLogger().handlers[0]
    assert handler.formatter is not None
    assert handler.formatter._fmt == "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    assert handler.formatter.datefmt == "%H:%M:%S"
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING


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

    class FakeRedisRuntime:
        managed = True

        @classmethod
        async def ensure(cls, _redis_url: str) -> FakeRedisRuntime:
            events.append("redis.ensure")
            return cls()

        async def close(self) -> None:
            events.append("redis.close")

    monkeypatch.setattr(cli, "build_app", lambda settings: FakeAppBundle())
    monkeypatch.setattr(cli, "build_scheduler", lambda settings: FakeSchedulerBundle())

    async def build_worker(_settings: object) -> FakeWorkerBundle:
        return FakeWorkerBundle()

    monkeypatch.setattr(cli, "build_worker", build_worker)
    monkeypatch.setattr(cli, "RedisRuntime", FakeRedisRuntime)

    await cli.run_all(
        SimpleNamespace(feishu_allow_from=(), redis_url="redis://localhost:6379/0")
    )

    assert events == [
        "redis.ensure",
        "worker.extensions",
        "app.start",
        "scheduler.run",
        "worker.run",
        "worker.close",
        "scheduler.close",
        "app.close",
        "redis.close",
    ]


async def test_run_all_can_embed_cli_in_same_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    finished = asyncio.Event()

    class FakeService:
        async def start(self) -> None:
            events.append("app.start")

        async def run_forever(self) -> None:
            await asyncio.Event().wait()

    class FakeBundle:
        service = FakeService()
        mcp_diagnostics: tuple[str, ...] = ()

        async def start_extensions(self) -> None:
            events.append("worker.extensions")

        async def close(self) -> None:
            events.append("close")

    class FakeRedisRuntime:
        managed = False

        @classmethod
        async def ensure(cls, _redis_url: str) -> FakeRedisRuntime:
            return cls()

        async def close(self) -> None:
            events.append("redis.close")

    async def fake_run_tui_async() -> None:
        events.append("tui.run")
        finished.set()

    monkeypatch.setattr(cli, "build_app", lambda _settings: FakeBundle())
    monkeypatch.setattr(cli, "build_scheduler", lambda _settings: FakeBundle())
    monkeypatch.setattr(cli, "build_worker", lambda _settings: _async_bundle(FakeBundle()))
    monkeypatch.setattr(cli, "RedisRuntime", FakeRedisRuntime)
    monkeypatch.setattr(cli, "run_tui_async", fake_run_tui_async)

    await cli.run_all(
        SimpleNamespace(feishu_allow_from=(), redis_url="redis://localhost:6379/0"),
        interactive=True,
    )

    assert finished.is_set()
    assert "tui.run" in events
    assert events[-1] == "redis.close"


async def test_external_terminal_shutdown_signal_closes_all_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    shutdown_event = asyncio.Event()

    class FakeService:
        async def start(self) -> None:
            return None

        async def run_forever(self) -> None:
            await asyncio.Event().wait()

    class FakeBundle:
        service = FakeService()
        mcp_diagnostics: tuple[str, ...] = ()

        async def start_extensions(self) -> None:
            return None

        async def close(self) -> None:
            events.append("close")

    class FakeRedisRuntime:
        managed = False

        @classmethod
        async def ensure(cls, _redis_url: str) -> FakeRedisRuntime:
            return cls()

        async def close(self) -> None:
            events.append("redis.close")

    async def fake_run_tui_async() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(cli, "build_app", lambda _settings: FakeBundle())
    monkeypatch.setattr(cli, "build_scheduler", lambda _settings: FakeBundle())
    monkeypatch.setattr(cli, "build_worker", lambda _settings: _async_bundle(FakeBundle()))
    monkeypatch.setattr(cli, "RedisRuntime", FakeRedisRuntime)
    monkeypatch.setattr(cli, "run_tui_async", fake_run_tui_async)

    task = asyncio.create_task(
        cli.run_all(
            SimpleNamespace(feishu_allow_from=(), redis_url="redis://localhost:6379/0"),
            interactive=True,
            shutdown_event=shutdown_event,
        )
    )
    await asyncio.sleep(0)
    shutdown_event.set()
    await task

    assert events == ["close", "close", "close", "redis.close"]


async def test_service_failure_is_propagated_after_orderly_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAppService:
        async def start(self) -> None:
            return None

    class FailingService:
        async def run_forever(self) -> None:
            raise RuntimeError("runner failed")

    class BlockingService:
        async def run_forever(self) -> None:
            await asyncio.Event().wait()

    class FakeBundle:
        mcp_diagnostics: tuple[str, ...] = ()

        def __init__(self, service: object) -> None:
            self.service = service

        async def start_extensions(self) -> None:
            return None

        async def close(self) -> None:
            events.append("close")

    class FakeRedisRuntime:
        managed = False

        @classmethod
        async def ensure(cls, _redis_url: str) -> FakeRedisRuntime:
            return cls()

        async def close(self) -> None:
            events.append("redis.close")

    monkeypatch.setattr(cli, "build_app", lambda _settings: FakeBundle(FakeAppService()))
    monkeypatch.setattr(
        cli,
        "build_scheduler",
        lambda _settings: FakeBundle(BlockingService()),
    )
    monkeypatch.setattr(
        cli,
        "build_worker",
        lambda _settings: _async_bundle(FakeBundle(FailingService())),
    )
    monkeypatch.setattr(cli, "RedisRuntime", FakeRedisRuntime)

    with pytest.raises(RuntimeError, match="runner failed"):
        await cli.run_all(
            SimpleNamespace(feishu_allow_from=(), redis_url="redis://localhost:6379/0"),
            shutdown_event=asyncio.Event(),
        )

    assert events == ["close", "close", "close", "redis.close"]


async def _async_bundle(bundle: object) -> object:
    return bundle
