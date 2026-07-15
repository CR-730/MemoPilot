"""Redis 中断通知镜像；SQLite 中断记录仍是事实源。"""

from __future__ import annotations

from typing import Protocol

from redis.asyncio import Redis


class InterruptSignalPort(Protocol):
    async def publish(self, run_id: str) -> None: ...

    async def pending(self, run_id: str) -> bool: ...

    async def clear(self, run_id: str) -> None: ...


class RedisInterruptSignal:
    def __init__(
        self,
        redis: Redis,
        *,
        namespace: str = "memopilot",
        ttl_seconds: int = 1800,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("中断通知 TTL 必须大于 0")
        self._redis = redis
        self._namespace = namespace
        self._ttl_seconds = ttl_seconds

    async def publish(self, run_id: str) -> None:
        await self._redis.set(self._key(run_id), "1", ex=self._ttl_seconds)

    async def pending(self, run_id: str) -> bool:
        return bool(await self._redis.exists(self._key(run_id)))

    async def clear(self, run_id: str) -> None:
        await self._redis.delete(self._key(run_id))

    def _key(self, run_id: str) -> str:
        return f"{self._namespace}:interrupt:run:{run_id}"


__all__ = ["InterruptSignalPort", "RedisInterruptSignal"]
