"""Redis 协调用户前台 Turn 与后台任务的最小边界。"""

from __future__ import annotations

from hashlib import sha256

from redis.asyncio import Redis


class RedisSessionCoordinator:
    """只保存会话抢占信号，不保存业务事实。"""

    def __init__(
        self,
        redis: Redis,
        *,
        namespace: str = "memopilot",
        preemption_ttl_seconds: int = 1800,
    ) -> None:
        if preemption_ttl_seconds <= 0:
            raise ValueError("Redis 会话协调 TTL 必须大于 0")
        self._redis = redis
        self._namespace = namespace
        self._preemption_ttl_seconds = preemption_ttl_seconds

    async def request_background_stop(self, session_key: str, *, reason: str) -> None:
        await self._redis.set(
            self.stop_key(session_key),
            reason,
            ex=self._preemption_ttl_seconds,
        )

    async def background_stop_requested(self, session_key: str) -> bool:
        return bool(await self._redis.exists(self.stop_key(session_key)))

    async def stop_reason(self, session_key: str) -> str | None:
        value = await self._redis.get(self.stop_key(session_key))
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else str(value)

    async def clear_background_stop(self, session_key: str) -> None:
        await self._redis.delete(self.stop_key(session_key))

    def stop_key(self, session_key: str) -> str:
        return f"{self._namespace}:session:{self._digest(session_key)}:preempt"

    @staticmethod
    def _digest(session_key: str) -> str:
        return sha256(session_key.encode()).hexdigest()


__all__ = ["RedisSessionCoordinator"]
