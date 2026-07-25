"""按需复用或托管本地 Redis 进程。"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from redis.asyncio import Redis


class RedisStartupError(RuntimeError):
    """Redis 在主服务启动前不可用。"""


class RedisProcess(Protocol):
    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


Ping = Callable[[str], Awaitable[bool]]
Launcher = Callable[[Path, str, int], Awaitable[RedisProcess]]
Sleeper = Callable[[float], Awaitable[None]]


class RedisRuntime:
    """记录 Redis 是否由当前 MemoPilot 进程启动，并管理其生命周期。"""

    def __init__(self, process: RedisProcess | None = None) -> None:
        self._process = process

    @property
    def managed(self) -> bool:
        return self._process is not None

    @classmethod
    async def ensure(
        cls,
        redis_url: str,
        *,
        executable: Path | None = None,
        ping: Ping | None = None,
        launch: Launcher | None = None,
        sleep: Sleeper = asyncio.sleep,
        attempts: int = 25,
    ) -> RedisRuntime:
        probe = ping or _ping
        if await probe(redis_url):
            return cls()

        parsed = urlsplit(redis_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        if host.casefold() not in {"localhost", "127.0.0.1", "::1"}:
            raise RedisStartupError(f"远程 Redis 不可用，无法自动启动本地替代服务: {host}:{port}")
        if parsed.username or parsed.password:
            raise RedisStartupError("本地 Redis 自动启动暂不接管带认证的连接地址")

        redis_executable = executable or _find_redis_server()
        if redis_executable is None:
            raise RedisStartupError(
                "未找到 redis-server；请安装 Redis 或设置 MEMOPILOT_REDIS_SERVER"
            )
        launcher = launch or _launch
        bind_host = "127.0.0.1" if host.casefold() == "localhost" else host
        try:
            process = await launcher(redis_executable, bind_host, port)
        except OSError as exc:
            raise RedisStartupError(f"redis-server 启动失败: {exc}") from exc

        runtime = cls(process)
        for _ in range(max(1, attempts)):
            if process.returncode is not None:
                raise RedisStartupError(
                    f"redis-server 启动后立即退出，退出码: {process.returncode}"
                )
            if await probe(redis_url):
                return runtime
            await sleep(0.2)
        await runtime.close()
        raise RedisStartupError("等待本地 Redis 就绪超时")

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            process.kill()
            await process.wait()


async def _ping(redis_url: str) -> bool:
    client = Redis.from_url(
        redis_url,
        socket_connect_timeout=0.25,
        socket_timeout=0.25,
    )
    try:
        return bool(await client.ping())
    except Exception:
        return False
    finally:
        await client.aclose()


def _find_redis_server() -> Path | None:
    configured = os.environ.get("MEMOPILOT_REDIS_SERVER", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(found) if (found := shutil.which("redis-server")) else None,
        Path(r"D:\Redis\redis-server.exe") if os.name == "nt" else None,
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


async def _launch(executable: Path, host: str, port: int) -> RedisProcess:
    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    return await asyncio.create_subprocess_exec(
        str(executable),
        "--bind",
        host,
        "--port",
        str(port),
        "--save",
        "",
        "--appendonly",
        "no",
        "--protected-mode",
        "yes",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        creationflags=creationflags,
    )


__all__ = ["RedisRuntime", "RedisStartupError"]
