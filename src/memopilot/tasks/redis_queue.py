"""Redis Streams 的 P0-P3 至少一次任务派发。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from redis.asyncio import Redis
from redis.exceptions import ResponseError


@dataclass(frozen=True, slots=True)
class PublishedJob:
    job_id: str
    kind: str
    priority: int
    session_key: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class QueueMessage:
    stream: str
    message_id: str
    job_id: str
    kind: str
    priority: int
    session_key: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class PendingEntry:
    message_id: str
    consumer_id: str


class RedisTaskQueue:
    """只保存可重建的队列副本，SQLite 仍是事实源。"""

    def __init__(
        self,
        redis: Redis,
        *,
        namespace: str = "memopilot",
        group: str = "memopilot-workers",
    ) -> None:
        self.redis = redis
        self.namespace = namespace
        self.group = group
        self.queued_job_ids_key = f"{namespace}:queued:job_ids"

    def stream_key(self, priority: int) -> str:
        if priority not in range(4):
            raise ValueError("priority 必须位于 0 到 3")
        return f"{self.namespace}:jobs:p{priority}"

    async def ensure_consumer_groups(self) -> None:
        for priority in range(4):
            try:
                await self.redis.xgroup_create(
                    self.stream_key(priority),
                    self.group,
                    id="0-0",
                    mkstream=True,
                )
            except ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise

    async def publish(self, job: PublishedJob) -> str:
        stream = self.stream_key(job.priority)
        message_id = await self.redis.xadd(
            stream,
            {
                "job_id": job.job_id,
                "kind": job.kind,
                "priority": str(job.priority),
                "session_key": job.session_key,
                "payload_json": job.payload_json,
            },
        )
        await self.redis.sadd(self.queued_job_ids_key, job.job_id)
        return _text(message_id)

    async def read_next(self, *, consumer_id: str) -> QueueMessage | None:
        for priority in range(4):
            stream = self.stream_key(priority)
            response = cast(
                list[
                    tuple[
                        str | bytes,
                        list[tuple[str | bytes, dict[object, object]]],
                    ]
                ],
                await self.redis.xreadgroup(
                    self.group,
                    consumer_id,
                    {stream: ">"},
                    count=1,
                ),
            )
            if not response:
                continue
            _, entries = response[0]
            message_id, fields = entries[0]
            return QueueMessage(
                stream=stream,
                message_id=_text(message_id),
                job_id=_field(fields, "job_id"),
                kind=_field(fields, "kind"),
                priority=int(_field(fields, "priority")),
                session_key=_field(fields, "session_key"),
                payload_json=_field(fields, "payload_json"),
            )
        return None

    async def acknowledge(self, message: QueueMessage) -> None:
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.xack(message.stream, self.group, message.message_id)
            pipeline.xdel(message.stream, message.message_id)
            pipeline.srem(self.queued_job_ids_key, message.job_id)
            await pipeline.execute()

    async def missing_mirrors(self, job_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not job_ids:
            return ()
        async with self.redis.pipeline(transaction=False) as pipeline:
            for job_id in job_ids:
                pipeline.sismember(self.queued_job_ids_key, job_id)
            present = await pipeline.execute()
        return tuple(job_id for job_id, exists in zip(job_ids, present, strict=True) if not exists)

    async def pending_entries(
        self,
        *,
        priority: int,
        min_idle_ms: int,
        after_message_id: str | None = None,
        count: int = 20,
    ) -> tuple[PendingEntry, ...]:
        minimum = "-" if after_message_id is None else f"({after_message_id}"
        raw = cast(
            list[dict[object, object]],
            await self.redis.xpending_range(
                self.stream_key(priority),
                self.group,
                min=minimum,
                max="+",
                count=count,
                idle=min_idle_ms,
            ),
        )
        return tuple(
            PendingEntry(
                message_id=_text(_mapping_field(item, "message_id")),
                consumer_id=_text(_mapping_field(item, "consumer")),
            )
            for item in raw
        )

    async def load_message(self, *, priority: int, message_id: str) -> QueueMessage | None:
        stream = self.stream_key(priority)
        entries = cast(
            list[tuple[str | bytes, dict[object, object]]],
            await self.redis.xrange(stream, min=message_id, max=message_id, count=1),
        )
        if not entries:
            return None
        entry_id, fields = entries[0]
        return _queue_message(stream, entry_id, fields)

    async def claim_pending(
        self,
        message: QueueMessage,
        *,
        consumer_id: str,
        min_idle_ms: int,
    ) -> QueueMessage | None:
        claimed = cast(
            list[tuple[str | bytes, dict[object, object]]],
            await self.redis.xclaim(
                message.stream,
                self.group,
                consumer_id,
                min_idle_ms,
                [message.message_id],
            ),
        )
        if not claimed:
            return None
        message_id, fields = claimed[0]
        return _queue_message(message.stream, message_id, fields)


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _field(fields: dict[object, object], name: str) -> str:
    value = fields.get(name)
    if value is None:
        value = fields.get(name.encode())
    if value is None:
        raise ValueError(f"Redis 任务缺少字段: {name}")
    return _text(value)


def _mapping_field(fields: dict[object, object], name: str) -> object:
    value = fields.get(name)
    if value is None:
        value = fields.get(name.encode())
    if value is None:
        raise ValueError(f"Redis Pending 记录缺少字段: {name}")
    return value


def _queue_message(
    stream: str,
    message_id: str | bytes,
    fields: dict[object, object],
) -> QueueMessage:
    return QueueMessage(
        stream=stream,
        message_id=_text(message_id),
        job_id=_field(fields, "job_id"),
        kind=_field(fields, "kind"),
        priority=int(_field(fields, "priority")),
        session_key=_field(fields, "session_key"),
        payload_json=_field(fields, "payload_json"),
    )


__all__ = ["PendingEntry", "PublishedJob", "QueueMessage", "RedisTaskQueue"]
