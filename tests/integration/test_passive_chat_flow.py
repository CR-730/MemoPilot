from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from memopilot.app.inbound import InboundBridge, OperationalInterruptController
from memopilot.channels.contracts import InboundMessage, MessageBus, SendReceipt
from memopilot.delivery.effects import EffectRepository
from memopilot.delivery.feishu import FinalResponseDispatcher
from memopilot.persistence.migrations import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema
from memopilot.runtime.engine import AgentRuntime
from memopilot.runtime.tools import ToolRegistry
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.tasks.interrupts import RedisInterruptSignal
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.outbox import OutboxDispatcher
from memopilot.tasks.redis_queue import PublishedJob, RedisTaskQueue
from memopilot.worker.service import WorkerService

NOW = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[Redis]:
    url = os.getenv("MEMOPILOT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
    client = Redis.from_url(url, decode_responses=True)
    await client.ping()
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


class _Provider:
    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        assert messages[-1].content == "你好"
        return ModelResponse(content="你好，我是 MemoPilot。", tool_calls=())


class _Transport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def send(
        self,
        chat_id: str,
        message: str,
        *,
        provider_uuid: str,
    ) -> SendReceipt:
        self.calls.append((chat_id, message, provider_uuid))
        return SendReceipt(message_id="om-reply-1")


def _repository(tmp_path: Path) -> OperationalRepository:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    return OperationalRepository(database)


@pytest.mark.asyncio
async def test_private_message_reaches_one_confirmed_agent_reply(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository = _repository(tmp_path)
    bus = MessageBus()
    inbound = InboundMessage(
        channel="feishu",
        sender="ou-user",
        chat_id="oc-chat",
        content="你好",
        timestamp=NOW,
        metadata={"event_id": "event-1", "message_id": "om-input-1"},
    )
    await bus.publish_inbound(inbound)
    accepted = await InboundBridge(repository).run_once(bus)
    duplicate = await InboundBridge(repository).handle(inbound)
    assert accepted.created is True
    assert duplicate.created is False

    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    assert await OutboxDispatcher(repository, queue, owner_id="app-1").dispatch_one(now=NOW)

    transport = _Transport()
    dispatcher = FinalResponseDispatcher(
        repository,
        EffectRepository(repository.database),
        transport,
        clock=lambda: NOW + timedelta(seconds=1),
    )
    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), ToolRegistry()),
        final_response_dispatcher=dispatcher,
        clock=lambda: NOW + timedelta(seconds=1),
    )
    worker = WorkerService(
        repository,
        queue,
        SessionLeaseManager(redis_client, repository, ttl=timedelta(seconds=2)),
        executor,
        owner_id="worker-1",
        clock=lambda: NOW + timedelta(seconds=1),
        heartbeat_interval=0.05,
    )

    assert await worker.run_once() is True
    assert await worker.run_once() is False

    job = repository.get_job(accepted.job_id)
    assert job is not None and job.state == "succeeded"
    assert repository.count("runs") == 1
    assert repository.count("outbound_effects") == 1
    assert transport.calls[0][:2] == ("oc-chat", "你好，我是 MemoPilot。")
    with connect_database(repository.database) as connection:
        effect = connection.execute(
            "SELECT state, message_id FROM outbound_effects"
        ).fetchone()
    assert effect is not None
    assert tuple(effect) == ("confirmed", "om-reply-1")
    assert transport.calls[0][2]

    await queue.publish(
        PublishedJob(
            accepted.job_id,
            "agent.turn",
            0,
            inbound.session_key,
            "{}",
        )
    )
    assert await worker.run_once() is True
    assert len(transport.calls) == 1
    pending = await redis_client.xpending(queue.stream_key(0), queue.group)
    assert pending["pending"] == 0


@pytest.mark.asyncio
async def test_stop_interrupts_worker_and_next_message_resumes_snapshot(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository = _repository(tmp_path)
    current = [NOW]
    first = await InboundBridge(repository).handle(
        InboundMessage(
            channel="feishu",
            sender="ou-user",
            chat_id="oc-chat",
            content="查天气后发邮件",
            timestamp=NOW,
            metadata={"event_id": "event-1", "message_id": "om-1"},
        )
    )
    queue = RedisTaskQueue(redis_client)
    signal = RedisInterruptSignal(redis_client)
    await queue.ensure_consumer_groups()
    assert await OutboxDispatcher(repository, queue, owner_id="app-1").dispatch_one(now=NOW)

    class _ResumableProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.messages: list[str] = []

        async def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSchema],
        ) -> ModelResponse:
            content = messages[-1].content or ""
            self.messages.append(content)
            if len(self.messages) == 1:
                self.started.set()
                await asyncio.Event().wait()
            return ModelResponse(content="已按补充要求继续完成", tool_calls=())

    provider = _ResumableProvider()
    transport = _Transport()
    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(provider, ToolRegistry()),
        final_response_dispatcher=FinalResponseDispatcher(
            repository,
            EffectRepository(repository.database),
            transport,
            clock=lambda: current[0],
        ),
        clock=lambda: current[0],
    )
    worker = WorkerService(
        repository,
        queue,
        SessionLeaseManager(redis_client, repository, ttl=timedelta(seconds=2)),
        executor,
        owner_id="worker-1",
        clock=lambda: current[0],
        heartbeat_interval=0.05,
        interrupt_poll_interval=0.01,
        interrupt_signal=signal,
    )

    running = asyncio.create_task(worker.run_once())
    await provider.started.wait()
    current[0] = NOW + timedelta(seconds=1)
    acknowledgement = await OperationalInterruptController(
        repository, signal=signal
    ).request_interrupt(
        InboundMessage(
            channel="feishu",
            sender="ou-user",
            chat_id="oc-chat",
            content="/stop",
            timestamp=current[0],
            metadata={"event_id": "stop-event", "message_id": "om-stop"},
        )
    )

    assert acknowledgement.message == "已收到停止请求，正在中断本轮任务。"
    assert await running is True
    assert repository.get_job(first.job_id).state == "cancelled"  # type: ignore[union-attr]
    assert transport.calls == []

    current[0] = NOW + timedelta(seconds=2)
    second = await InboundBridge(repository).handle(
        InboundMessage(
            channel="feishu",
            sender="ou-user",
            chat_id="oc-chat",
            content="改成发给小王",
            timestamp=current[0],
            metadata={"event_id": "event-2", "message_id": "om-2"},
        )
    )
    assert await OutboxDispatcher(repository, queue, owner_id="app-1").dispatch_one(
        now=current[0]
    )
    assert await worker.run_once() is True

    assert repository.get_job(second.job_id).state == "succeeded"  # type: ignore[union-attr]
    assert "上一轮任务" in provider.messages[-1]
    assert "查天气后发邮件" in provider.messages[-1]
    assert "改成发给小王" in provider.messages[-1]
    with connect_database(repository.database) as connection:
        consumed = connection.execute(
            "SELECT consumed_at FROM turn_interrupt_snapshots"
        ).fetchone()
    assert consumed is not None and consumed["consumed_at"] is not None


@pytest.mark.asyncio
async def test_message_blocked_by_session_lease_is_reclaimed_after_release(
    tmp_path: Path,
    redis_client: Redis,
) -> None:
    repository = _repository(tmp_path)
    accepted = await InboundBridge(repository).handle(
        InboundMessage(
            channel="feishu",
            sender="ou-user",
            chat_id="oc-chat",
            content="你好",
            timestamp=NOW,
            metadata={"event_id": "event-pending", "message_id": "om-pending"},
        )
    )
    queue = RedisTaskQueue(redis_client)
    await queue.ensure_consumer_groups()
    assert await OutboxDispatcher(repository, queue, owner_id="app-1").dispatch_one(now=NOW)
    leases = SessionLeaseManager(redis_client, repository, ttl=timedelta(seconds=2))
    blocker = await leases.acquire("feishu:oc-chat", owner_id="worker-a", now=NOW)
    assert blocker is not None

    transport = _Transport()
    executor = RuntimeJobExecutor(
        repository,
        AgentRuntime(_Provider(), ToolRegistry()),
        final_response_dispatcher=FinalResponseDispatcher(
            repository,
            EffectRepository(repository.database),
            transport,
            clock=lambda: NOW + timedelta(seconds=1),
        ),
        clock=lambda: NOW + timedelta(seconds=1),
    )
    worker = WorkerService(
        repository,
        queue,
        leases,
        executor,
        owner_id="worker-b",
        clock=lambda: NOW + timedelta(seconds=1),
        heartbeat_interval=0.05,
        pending_min_idle=timedelta(milliseconds=10),
        stale_heartbeat=timedelta(seconds=1),
        pending_reclaim_every=1,
    )

    assert await worker.run_once() is False
    assert transport.calls == []
    assert await leases.release(blocker) is True
    await asyncio.sleep(0.02)
    await queue.publish(PublishedJob("unrelated", "agent.turn", 0, "feishu:other", "{}"))

    assert await worker.run_once() is True
    assert repository.get_job(accepted.job_id).state == "succeeded"  # type: ignore[union-attr]
    assert len(transport.calls) == 1
