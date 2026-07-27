"""Redis 协调用户前台 Turn 与后台任务的最小边界。"""

from __future__ import annotations

from hashlib import sha256

from redis.asyncio import Redis

_COMPARE_DELETE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisSessionCoordinator:
    """只保存会话占用和抢占信号，不保存业务事实。"""

    def __init__(
        self,
        redis: Redis,
        *,
        namespace: str = "memopilot",
        user_turn_ttl_seconds: int = 1800,
        preemption_ttl_seconds: int = 1800,
    ) -> None:
        if user_turn_ttl_seconds <= 0 or preemption_ttl_seconds <= 0:
            raise ValueError("Redis 会话协调 TTL 必须大于 0")
        self._redis = redis
        self._namespace = namespace
        self._user_turn_ttl_seconds = user_turn_ttl_seconds
        self._preemption_ttl_seconds = preemption_ttl_seconds

    async def begin_user_turn(self, session_key: str, *, turn_id: str) -> bool:
        return bool(
            await self._redis.set(
                self._user_key(session_key),
                turn_id,
                ex=self._user_turn_ttl_seconds,
            )
        )

    async def renew_user_turn(self, session_key: str, *, turn_id: str) -> bool:
        key = self._user_key(session_key)
        current = await self._redis.get(key)
        if current != turn_id:
            return False
        return bool(await self._redis.expire(key, self._user_turn_ttl_seconds))

    async def end_user_turn(self, session_key: str, *, turn_id: str) -> bool:
        deleted = await self._redis.eval(
            _COMPARE_DELETE,
            1,
            self._user_key(session_key),
            turn_id,
        )
        return int(deleted) == 1

    async def user_turn_active(self, session_key: str) -> bool:
        return bool(await self._redis.exists(self._user_key(session_key)))

    async def request_background_stop(self, session_key: str, *, reason: str) -> None:
        await self._redis.set(
            self._stop_key(session_key),
            reason,
            ex=self._preemption_ttl_seconds,
        )

    async def background_stop_requested(self, session_key: str) -> bool:
        return bool(await self._redis.exists(self._stop_key(session_key)))

    async def session_busy(self, session_key: str) -> bool:
        digest = self._digest(session_key)
        user_key = self._user_key(session_key)
        lease_key = f"{self._namespace}:lease:{digest}"
        return bool(await self._redis.exists(user_key, lease_key))

    async def clear_background_stop(self, session_key: str) -> None:
        await self._redis.delete(self._stop_key(session_key))

    def _user_key(self, session_key: str) -> str:
        return f"{self._namespace}:session:{self._digest(session_key)}:user-turn"

    def _stop_key(self, session_key: str) -> str:
        return f"{self._namespace}:session:{self._digest(session_key)}:preempt"

    @staticmethod
    def _digest(session_key: str) -> str:
        return sha256(session_key.encode()).hexdigest()


__all__ = ["RedisSessionCoordinator"]
