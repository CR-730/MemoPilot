"""Redis 临时租约与 SQLite 单调 fencing epoch。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from redis.asyncio import Redis

from memopilot.tasks.operational import OperationalRepository

_COMPARE_PEXPIRE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

_COMPARE_DELETE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

_FINALIZE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('PSETEX', KEYS[1], ARGV[3], ARGV[2])
  return 1
end
return 0
"""


@dataclass(frozen=True, slots=True)
class SessionLease:
    session_key: str
    owner_id: str
    epoch: int
    redis_key: str
    redis_value: str


class SessionLeaseManager:
    def __init__(
        self,
        redis: Redis,
        repository: OperationalRepository,
        *,
        ttl: timedelta = timedelta(seconds=30),
        namespace: str = "memopilot",
    ) -> None:
        if ttl.total_seconds() <= 0:
            raise ValueError("lease TTL 必须大于 0")
        self.redis = redis
        self.repository = repository
        self.ttl_ms = int(ttl.total_seconds() * 1000)
        self.namespace = namespace

    async def acquire(
        self,
        session_key: str,
        *,
        owner_id: str,
        now: datetime,
    ) -> SessionLease | None:
        key = self.lease_key(session_key)
        token = uuid4().hex
        provisional = f"{owner_id}|provisional|{token}"
        acquired = await self.redis.set(key, provisional, nx=True, px=self.ttl_ms)
        if not acquired:
            return None
        try:
            epoch = self.repository.allocate_fence(session_key, owner_id=owner_id, now=now)
        except Exception:
            await self.redis.eval(_COMPARE_DELETE, 1, key, provisional)
            raise
        final = f"{owner_id}|epoch|{epoch}"
        finalized = await self.redis.eval(
            _FINALIZE,
            1,
            key,
            provisional,
            final,
            self.ttl_ms,
        )
        if int(finalized) != 1:
            await self.redis.eval(_COMPARE_DELETE, 1, key, provisional)
            return None
        return SessionLease(session_key, owner_id, epoch, key, final)

    async def renew(self, lease: SessionLease, *, now: datetime | None = None) -> bool:
        renewed = await self.redis.eval(
            _COMPARE_PEXPIRE,
            1,
            lease.redis_key,
            lease.redis_value,
            self.ttl_ms,
        )
        if int(renewed) != 1:
            return False
        heartbeat_at = now or datetime.now(UTC)
        return self.repository.heartbeat_fence(lease, now=heartbeat_at)

    async def release(self, lease: SessionLease) -> bool:
        deleted = await self.redis.eval(
            _COMPARE_DELETE,
            1,
            lease.redis_key,
            lease.redis_value,
        )
        return int(deleted) == 1

    async def is_absent(self, session_key: str) -> bool:
        return not bool(await self.redis.exists(self.lease_key(session_key)))

    def lease_key(self, session_key: str) -> str:
        digest = sha256(session_key.encode()).hexdigest()
        return f"{self.namespace}:lease:{digest}"


__all__ = ["SessionLease", "SessionLeaseManager"]
