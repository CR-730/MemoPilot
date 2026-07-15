"""阶段 3 的 App、Worker 与 Effect 操作装配入口。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

from redis.asyncio import Redis

from memopilot.app.inbound import InboundBridge, OperationalInterruptController
from memopilot.app.service import AppService
from memopilot.channels.base import AttachmentStore, SessionIdentityIndex
from memopilot.channels.contracts import MessageBus
from memopilot.channels.feishu import FeishuChannel
from memopilot.config import MemoPilotSettings
from memopilot.delivery.effects import EffectRepository
from memopilot.delivery.feishu import FinalResponseDispatcher
from memopilot.delivery.reconciliation import EffectReconciliationService
from memopilot.persistence.migrations import DatabaseKind, migrate_database
from memopilot.runtime.engine import AgentRuntime
from memopilot.runtime.providers import OpenAICompatibleProvider
from memopilot.runtime.tools import ToolRegistry
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.tasks.interrupts import RedisInterruptSignal
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.outbox import OutboxDispatcher
from memopilot.tasks.redis_queue import RedisTaskQueue
from memopilot.worker.service import WorkerService


@dataclass(slots=True)
class AppBundle:
    service: AppService
    redis: Redis

    async def close(self) -> None:
        await self.service.stop()
        await self.redis.aclose()


@dataclass(slots=True)
class WorkerBundle:
    service: WorkerService
    transport: FeishuChannel
    redis: Redis

    async def close(self) -> None:
        await self.transport.stop()
        await self.redis.aclose()


@dataclass(slots=True)
class EffectBundle:
    service: EffectReconciliationService
    transport: FeishuChannel
    redis: Redis

    async def close(self) -> None:
        await self.transport.stop()
        await self.redis.aclose()


def build_app(settings: MemoPilotSettings) -> AppBundle:
    settings.validate_app_ready()
    repository = _repository(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    queue = RedisTaskQueue(redis)
    signal = RedisInterruptSignal(redis)
    bus = MessageBus()
    channel = _feishu_channel(
        settings,
        repository,
        bus=bus,
        interrupt_controller=OperationalInterruptController(repository, signal=signal),
    )
    service = AppService(
        channel=channel,
        bus=bus,
        bridge=InboundBridge(repository),
        queue=queue,
        outbox=OutboxDispatcher(repository, queue, owner_id=f"app-{uuid4().hex[:8]}"),
    )
    return AppBundle(service, redis)


def build_worker(settings: MemoPilotSettings) -> WorkerBundle:
    settings.validate_worker_ready()
    repository = _repository(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    queue = RedisTaskQueue(redis)
    leases = SessionLeaseManager(
        redis,
        repository,
        ttl=timedelta(seconds=settings.lease_ttl_seconds),
    )
    transport = _feishu_channel(settings, repository, bus=MessageBus())
    provider = OpenAICompatibleProvider.from_deepseek_credentials(
        api_key=settings.chat_api_key.get_secret_value(),
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_retry_limit,
        timeout_seconds=settings.llm_timeout_seconds,
        thinking_enabled=settings.llm_thinking_enabled,
    )
    runtime = AgentRuntime(
        provider,
        ToolRegistry(),
        max_iterations=settings.llm_max_iterations,
    )
    dispatcher = FinalResponseDispatcher(
        repository,
        EffectRepository(repository.database),
        transport,
    )
    executor = RuntimeJobExecutor(
        repository,
        runtime,
        final_response_dispatcher=dispatcher,
    )
    service = WorkerService(
        repository,
        queue,
        leases,
        executor,
        owner_id=f"worker-{uuid4().hex[:8]}",
        heartbeat_interval=settings.lease_heartbeat_seconds,
        pending_min_idle=timedelta(seconds=settings.reclaim_idle_seconds),
        stale_heartbeat=timedelta(seconds=settings.reclaim_idle_seconds),
        interrupt_signal=RedisInterruptSignal(redis),
    )
    return WorkerBundle(service, transport, redis)


def build_effects(settings: MemoPilotSettings) -> EffectBundle:
    settings.validate_worker_ready()
    repository = _repository(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    leases = SessionLeaseManager(
        redis,
        repository,
        ttl=timedelta(seconds=settings.lease_ttl_seconds),
    )
    transport = _feishu_channel(settings, repository, bus=MessageBus())
    effects = EffectRepository(repository.database)
    service = EffectReconciliationService(
        repository,
        effects,
        leases,
        FinalResponseDispatcher(repository, effects, transport),
        owner_id=f"operator-{uuid4().hex[:8]}",
    )
    return EffectBundle(service, transport, redis)


def _repository(settings: MemoPilotSettings) -> OperationalRepository:
    migrate_database(
        settings.operational_database,
        DatabaseKind.OPERATIONAL,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )
    return OperationalRepository(
        settings.operational_database,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )


def _feishu_channel(
    settings: MemoPilotSettings,
    repository: OperationalRepository,
    *,
    bus: MessageBus,
    interrupt_controller: OperationalInterruptController | None = None,
) -> FeishuChannel:
    return FeishuChannel(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret.get_secret_value(),
        bus=bus,
        identity_index=SessionIdentityIndex(
            repository,
            channel=settings.feishu_channel_name,
        ),
        attachment_store=AttachmentStore(settings.uploads_dir),
        allow_from=settings.feishu_allow_from,
        interrupt_controller=interrupt_controller,
        channel_name=settings.feishu_channel_name,
    )


__all__ = [
    "AppBundle",
    "EffectBundle",
    "WorkerBundle",
    "build_app",
    "build_effects",
    "build_worker",
]
