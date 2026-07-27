from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import memopilot.redis_runtime as redis_runtime
from memopilot.redis_runtime import RedisRuntime, RedisStartupError


class FakeProcess:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -1

    async def wait(self) -> int:
        return self.returncode or 0


@pytest.mark.asyncio
async def test_existing_local_redis_is_reused_and_not_stopped() -> None:
    launched = False

    async def ping(_url: str) -> bool:
        return True

    async def launch(_executable: Path, _host: str, _port: int) -> FakeProcess:
        nonlocal launched
        launched = True
        return FakeProcess()

    runtime = await RedisRuntime.ensure(
        "redis://localhost:6379/0",
        ping=ping,
        launch=launch,
    )
    await runtime.close()
    assert runtime.managed is False
    assert launched is False


@pytest.mark.asyncio
async def test_missing_local_redis_is_started_and_stopped() -> None:
    process = FakeProcess()
    ping_results = iter([False, False, True])
    launch_args: tuple[Path, str, int] | None = None

    async def ping(_url: str) -> bool:
        return next(ping_results)

    async def launch(executable: Path, host: str, port: int) -> FakeProcess:
        nonlocal launch_args
        launch_args = (executable, host, port)
        return process

    runtime = await RedisRuntime.ensure(
        "redis://127.0.0.1:6380/0",
        executable=Path("redis-server.exe"),
        ping=ping,
        launch=launch,
        sleep=lambda _seconds: _completed_sleep(),
    )
    assert runtime.managed is True
    await runtime.close()
    assert runtime.managed is False
    assert launch_args == (Path("redis-server.exe"), "127.0.0.1", 6380)
    assert process.terminated is True


@pytest.mark.asyncio
async def test_remote_redis_is_never_started() -> None:
    async def ping(_url: str) -> bool:
        return False

    async def launch(_executable: Path, _host: str, _port: int) -> FakeProcess:
        raise AssertionError("远程 Redis 不应启动本地进程")

    with pytest.raises(RedisStartupError, match="远程 Redis"):
        await RedisRuntime.ensure(
            "redis://redis.example.com:6379/0",
            ping=ping,
            launch=launch,
        )


@pytest.mark.asyncio
async def test_local_redis_startup_failure_is_reported() -> None:
    process = FakeProcess(returncode=1)

    async def ping(_url: str) -> bool:
        return False

    async def launch(_executable: Path, _host: str, _port: int) -> FakeProcess:
        return process

    with pytest.raises(RedisStartupError, match="启动后立即退出"):
        await RedisRuntime.ensure(
            "redis://localhost:6379/0",
            executable=Path("redis-server.exe"),
            ping=ping,
            launch=launch,
        )


@pytest.mark.asyncio
async def test_managed_redis_uses_user_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(redis_runtime, "_redis_runtime_directory", lambda: tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    await redis_runtime._launch(Path("redis-server.exe"), "127.0.0.1", 6379)

    assert captured["kwargs"]["cwd"] == str(tmp_path)  # type: ignore[index]


async def _completed_sleep() -> None:
    return None
