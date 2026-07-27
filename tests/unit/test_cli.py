from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot import cli


def test_cli_default_workspace_uses_isolated_user_workspace() -> None:
    assert cli._default_workspace() == Path.home() / ".memopilot" / "memopilot-workspace"


def test_configure_logging_hides_http_handshake_noise() -> None:
    cli._configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


class _RedisRuntime:
    managed = False

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _AppRuntime:
    mcp_diagnostics: tuple[str, ...] = ()

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def run_forever(self) -> None:
        if self.fail:
            raise RuntimeError("service failed")
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


async def test_run_all_starts_and_closes_single_app_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _RedisRuntime()
    runtime = _AppRuntime()
    shutdown = asyncio.Event()

    async def ensure(_url: str) -> _RedisRuntime:
        return redis

    async def build(_settings: object) -> _AppRuntime:
        return runtime

    monkeypatch.setattr(cli.RedisRuntime, "ensure", ensure)
    monkeypatch.setattr(cli, "build_app_runtime", build)
    task = asyncio.create_task(
        cli.run_all(
            SimpleNamespace(redis_url="redis://local", feishu_allow_from=()),
            shutdown_event=shutdown,
        )
    )
    await asyncio.sleep(0)
    shutdown.set()
    await task

    assert runtime.started is True
    assert runtime.closed is True
    assert redis.closed is True


async def test_service_failure_is_propagated_after_orderly_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _RedisRuntime()
    runtime = _AppRuntime(fail=True)

    async def ensure(_url: str) -> _RedisRuntime:
        return redis

    async def build(_settings: object) -> _AppRuntime:
        return runtime

    monkeypatch.setattr(cli.RedisRuntime, "ensure", ensure)
    monkeypatch.setattr(cli, "build_app_runtime", build)

    with pytest.raises(RuntimeError, match="service failed"):
        await cli.run_all(SimpleNamespace(redis_url="redis://local", feishu_allow_from=()))

    assert runtime.closed is True
    assert redis.closed is True
