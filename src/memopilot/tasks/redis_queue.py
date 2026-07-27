"""Redis Streams 的 P0-P3 至少一次任务派发。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from memopilot.tasks.background import BackgroundTask

_PUBLISH_ONCE = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return ''
end
redis.call('SET', KEYS[1], '1', 'EX', ARGV[1])
local message_id = redis.call(
  'XADD', KEYS[2], '*',
  'task_id', ARGV[2],
  'kind', ARGV[3],
  'priority', ARGV[4],
  'session_key', ARGV[5],
  'payload_json', ARGV[6]
)
redis.call('SADD', KEYS[3], ARGV[2])
return message_id
"""


@dataclass(frozen=True, slots=True)
class PublishedTask:
    task_id: str
    kind: str
    priority: int
    session_key: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class QueueMessage:
    stream: str
    message_id: str
    task_id: str
    kind: str
    priority: int
    session_key: str
    payload_json: str

@dataclass(frozen=True, slots=True)
class PendingEntry:
    message_id: str
    consumer_id: str


class RedisTaskQueue:
    """后台任务的优先级队列与短期执行状态。"""

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
        self.queued_task_ids_key = f"{namespace}:queued:task_ids"

    def stream_key(self, priority: int) -> str:
        if priority not in range(4):
            raise ValueError("priority 必须位于 0 到 3")
        return f"{self.namespace}:tasks:p{priority}"

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

    async def publish(self, task: PublishedTask) -> str:
        stream = self.stream_key(task.priority)
        message_id = await self.redis.xadd(
            stream,
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "priority": str(task.priority),
                "session_key": task.session_key,
                "payload_json": task.payload_json,
            },
        )
        await self.redis.sadd(self.queued_task_ids_key, task.task_id)
        return _text(message_id)

    async def publish_task_once(
        self, task: BackgroundTask, *, ttl_seconds: int = 86400
    ) -> str | None:
        """以任务 ID 做 Redis 幂等，避免 Scheduler 重启重复投递同一时间桶。"""
        key = f"{self.namespace}:task:{task.task_id}"
        result = await self.redis.eval(
            _PUBLISH_ONCE,
            3,
            key,
            self.stream_key(task.priority),
            self.queued_task_ids_key,
            ttl_seconds,
            task.task_id,
            task.kind,
            task.priority,
            task.session_key,
            task.payload_json,
        )
        message_id = _text(result)
        return message_id or None

    async def read_next(self, *, consumer_id: str) -> QueueMessage | None:
        try:
            return await self._read_next_from_existing_groups(consumer_id=consumer_id)
        except ResponseError as exc:
            if "NOGROUP" not in str(exc):
                raise
            await self.ensure_consumer_groups()
            return await self._read_next_from_existing_groups(consumer_id=consumer_id)

    async def _read_next_from_existing_groups(
        self,
        *,
        consumer_id: str,
    ) -> QueueMessage | None:
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
                task_id=_field(fields, "task_id"),
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
            pipeline.srem(self.queued_task_ids_key, message.task_id)
            await pipeline.execute()

    async def acknowledge_requeued(self, message: QueueMessage) -> None:
        """移除当前投递，但保留同业务任务的 Redis 镜像标记。"""
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.xack(message.stream, self.group, message.message_id)
            pipeline.xdel(message.stream, message.message_id)
            await pipeline.execute()

    async def missing_tasks(self, task_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not task_ids:
            return ()
        async with self.redis.pipeline(transaction=False) as pipeline:
            for task_id in task_ids:
                pipeline.sismember(self.queued_task_ids_key, task_id)
            present = await pipeline.execute()
        return tuple(
            task_id
            for task_id, exists in zip(task_ids, present, strict=True)
            if not exists
        )

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
        task_id=_field(fields, "task_id"),
        kind=_field(fields, "kind"),
        priority=int(_field(fields, "priority")),
        session_key=_field(fields, "session_key"),
        payload_json=_field(fields, "payload_json"),
    )


__all__ = ["PendingEntry", "PublishedTask", "QueueMessage", "RedisTaskQueue"]
