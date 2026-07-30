"""基于 Redis Streams 的 P0—P3 优先级任务队列。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from memopilot.bus.events import InboundMessage
from memopilot.tasks.agent_task import AgentTask

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

_PUBLISH_INBOUND = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return ''
end
redis.call('SET', KEYS[1], '1', 'EX', ARGV[1])
local message_id = redis.call(
  'XADD', KEYS[2], '*',
  'task_id', ARGV[2],
  'kind', 'passive.turn',
  'priority', '0',
  'session_key', ARGV[3],
  'payload_json', ARGV[4]
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


class RedisTaskQueue:
    """提供幂等发布、Pending 重放、读取和 ACK。"""

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

    async def publish_task_once(self, task: AgentTask, *, ttl_seconds: int = 86400) -> str | None:
        """按稳定任务 ID 幂等发布，避免生产器重启造成重复投递。"""
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

    async def publish_inbound(
        self,
        message: InboundMessage,
        *,
        ttl_seconds: int = 86400,
    ) -> str | None:
        identity = str(message.metadata.get("message_id") or "").strip()
        if not identity:
            identity = f"{message.session_key}:{message.timestamp.isoformat()}:{message.content}"
        task_id = str(uuid5(NAMESPACE_URL, f"memopilot:passive:{identity}"))
        payload_json = json.dumps(
            {
                "channel": message.channel,
                "sender": message.sender,
                "chat_id": message.chat_id,
                "content": message.content,
                "timestamp": message.timestamp.astimezone(UTC).isoformat(),
                "media": list(message.media),
                "metadata": dict(message.metadata),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result = await self.redis.eval(
            _PUBLISH_INBOUND,
            3,
            f"{self.namespace}:task:{task_id}",
            self.stream_key(0),
            self.queued_task_ids_key,
            ttl_seconds,
            task_id,
            message.session_key,
            payload_json,
        )
        message_id = _text(result)
        return message_id or None

    async def read_next(self, *, consumer_id: str) -> QueueMessage | None:
        try:
            return await self._read_from_existing_groups(consumer_id=consumer_id, entry_id=">")
        except ResponseError as exc:
            if "NOGROUP" not in str(exc):
                raise
            await self.ensure_consumer_groups()
            return await self._read_from_existing_groups(consumer_id=consumer_id, entry_id=">")

    async def read_pending(self, *, consumer_id: str) -> QueueMessage | None:
        """读取固定 consumer 自己尚未 ACK 的 Pending 消息。"""
        try:
            return await self._read_from_existing_groups(consumer_id=consumer_id, entry_id="0")
        except ResponseError as exc:
            if "NOGROUP" not in str(exc):
                raise
            await self.ensure_consumer_groups()
            return await self._read_from_existing_groups(consumer_id=consumer_id, entry_id="0")

    async def read_priority(
        self, priority: int, *, consumer_id: str, entry_id: str = ">"
    ) -> QueueMessage | None:
        """读取指定优先级的新消息，用于运行中的 P0 抢占检查。"""
        stream = self.stream_key(priority)
        response = cast(
            list[tuple[str | bytes, list[tuple[str | bytes, dict[object, object]]]]],
            await self.redis.xreadgroup(self.group, consumer_id, {stream: entry_id}, count=1),
        )
        if not response:
            return None
        _, entries = response[0]
        message_id, fields = entries[0]
        return _queue_message(stream, message_id, fields)

    async def _read_from_existing_groups(
        self,
        *,
        consumer_id: str,
        entry_id: str,
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
                    {stream: entry_id},
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

    async def missing_tasks(self, task_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not task_ids:
            return ()
        async with self.redis.pipeline(transaction=False) as pipeline:
            for task_id in task_ids:
                pipeline.sismember(self.queued_task_ids_key, task_id)
            present = await pipeline.execute()
        return tuple(
            task_id for task_id, exists in zip(task_ids, present, strict=True) if not exists
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


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _field(fields: dict[object, object], name: str) -> str:
    value = fields.get(name)
    if value is None:
        value = fields.get(name.encode())
    if value is None:
        raise ValueError(f"Redis 任务缺少字段: {name}")
    return _text(value)


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


__all__ = ["PublishedTask", "QueueMessage", "RedisTaskQueue"]
