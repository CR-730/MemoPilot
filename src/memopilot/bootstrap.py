"""App、Worker、Effect 与分层记忆的集中装配入口。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import cast
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
from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.contracts import EmbeddingProvider
from memopilot.memory.engine import LayeredMemoryEngine
from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.memory.memorizer import MemoryMemorizer
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.post_response import OperationalPostResponseService, PostResponseMemoryWorker
from memopilot.memory.providers import (
    ChatConsolidationExtractor,
    ChatHypothesisProvider,
    ChatImplicitMemoryExtractor,
    ChatOptimizerModel,
    ChatPostResponseModel,
    ChatProcedureTagger,
    ChatRecentContextCompressor,
    OpenAIEmbeddingProvider,
)
from memopilot.memory.retrieval import MemoryRetriever, RetrievalStore
from memopilot.memory.scheduler import MemoryMaintenanceScheduler
from memopilot.memory.store import MemoryStore
from memopilot.memory.tools import (
    build_forget_memory_tool,
    build_memorize_tool,
    build_recall_memory_tool,
)
from memopilot.memory.vectorization import VectorizationService
from memopilot.memory.worker import MemoryJobRouter
from memopilot.persistence.memory_metadata import EmbeddingIdentity, ensure_embedding_identity
from memopilot.persistence.migrations import (
    DatabaseKind,
    migrate_all_databases,
    migrate_database,
)
from memopilot.runtime.engine import AgentRuntime
from memopilot.runtime.providers import ChatProvider, OpenAICompatibleProvider
from memopilot.runtime.tools import Tool, ToolRegistry
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
    runtime: RuntimeBundle

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


@dataclass(slots=True)
class SchedulerBundle:
    service: MemoryMaintenanceScheduler


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    repository: OperationalRepository
    tools: ToolRegistry
    memory_engine: LayeredMemoryEngine
    memory_jobs: MemoryJobRouter
    runtime: AgentRuntime
    executor: RuntimeJobExecutor


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


def build_scheduler(settings: MemoPilotSettings) -> SchedulerBundle:
    repository = _repository(settings)
    return SchedulerBundle(
        MemoryMaintenanceScheduler(
            repository,
            enabled=settings.memory_optimizer_enabled,
            interval=timedelta(seconds=settings.memory_optimizer_interval_seconds),
        )
    )


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
    provider = _chat_provider(settings)
    dispatcher = FinalResponseDispatcher(
        repository,
        EffectRepository(repository.database),
        transport,
    )
    runtime_bundle = build_runtime_bundle(
        settings,
        chat_provider=provider,
        final_response_dispatcher=dispatcher,
        repository=repository,
    )
    service = WorkerService(
        repository,
        queue,
        leases,
        runtime_bundle.executor,
        owner_id=f"worker-{uuid4().hex[:8]}",
        heartbeat_interval=settings.lease_heartbeat_seconds,
        pending_min_idle=timedelta(seconds=settings.reclaim_idle_seconds),
        stale_heartbeat=timedelta(seconds=settings.reclaim_idle_seconds),
        short_term_message_limit=settings.memory_short_term_message_limit,
        interrupt_signal=RedisInterruptSignal(redis),
    )
    return WorkerBundle(service, transport, redis, runtime_bundle)


def build_effects(settings: MemoPilotSettings) -> EffectBundle:
    settings.validate_effects_ready()
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


def build_runtime_bundle(
    settings: MemoPilotSettings,
    *,
    chat_provider: ChatProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    tools: Iterable[Tool] = (),
    final_response_dispatcher: FinalResponseDispatcher | None = None,
    repository: OperationalRepository | None = None,
) -> RuntimeBundle:
    """从类型化配置创建 Worker 使用的 Agent 与分层记忆链路。"""
    configured_tools = tuple(tools)
    migrate_all_databases(settings)
    operational = repository or OperationalRepository(
        settings.operational_database,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )
    provider = chat_provider or _chat_provider(settings)
    embedding_provider = embedder or OpenAIEmbeddingProvider(
        api_key=settings.embedding_api_key.get_secret_value(),
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )
    ensure_embedding_identity(
        settings.memory_database,
        EmbeddingIdentity(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        ),
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )

    markdown = MarkdownMemoryStore(settings.memory_dir)
    store = MemoryStore(
        settings.memory_database,
        dimension=settings.embedding_dimension,
        hotness_alpha=settings.memory_hotness_alpha,
        hotness_half_life_days=settings.memory_hotness_half_life_days,
    )
    retriever = MemoryRetriever(
        cast(RetrievalStore, store),
        embedding_provider,
        score_threshold=settings.memory_score_threshold,
        score_thresholds=settings.memory_score_thresholds,
        embed_timeout_seconds=settings.memory_embed_timeout_seconds,
        procedure_guard_enabled=settings.memory_procedure_guard_enabled,
        inject_max_chars=settings.memory_inject_max_chars,
        inject_max_forced=settings.memory_inject_max_forced,
        inject_max_procedure_preference=settings.memory_inject_max_procedure_preference,
        inject_max_event_profile=settings.memory_inject_max_event_profile,
    )
    memory_engine = LayeredMemoryEngine(
        retriever,
        hypothesis_provider=ChatHypothesisProvider(provider),
    )
    memorizer = MemoryMemorizer(
        store,
        embedding_provider,
        procedure_tagger=ChatProcedureTagger(
            provider,
            allowed_tools={
                *(tool.name for tool in configured_tools),
                "recall_memory",
                "memorize",
                "forget_memory",
            },
            allowed_skills=set(),
        ),
    )
    registry = ToolRegistry(configured_tools)
    registry.register(build_recall_memory_tool(memory_engine))
    registry.register(build_memorize_tool(memorizer))
    registry.register(build_forget_memory_tool(store))
    runtime = AgentRuntime(
        provider,
        registry,
        max_iterations=settings.llm_max_iterations,
        memory_engine=memory_engine,
        memory_profile=markdown,
    )
    memory_jobs = MemoryJobRouter(
        ConsolidationService(
            settings.operational_database,
            markdown,
            ChatConsolidationExtractor(provider, markdown),
            recent_context=ChatRecentContextCompressor(provider),
            keep_count=settings.memory_consolidation_keep_count,
            min_new_messages=settings.memory_consolidation_min_new_messages,
        ),
        VectorizationService(
            settings.operational_database,
            store,
            embedding_provider,
            memorizer=memorizer,
            implicit_extractor=ChatImplicitMemoryExtractor(provider),
            display_timezone=settings.display_timezone,
        ),
        MemoryOptimizer(markdown, ChatOptimizerModel(provider)),
        operational,
        post_response=OperationalPostResponseService(
            settings.operational_database,
            PostResponseMemoryWorker(
                store,
                retriever,
                ChatPostResponseModel(provider),
            ),
        ),
    )
    executor = RuntimeJobExecutor(
        operational,
        runtime,
        final_response_dispatcher=final_response_dispatcher,
        memory_jobs=memory_jobs,
    )
    return RuntimeBundle(
        repository=operational,
        tools=registry,
        memory_engine=memory_engine,
        memory_jobs=memory_jobs,
        runtime=runtime,
        executor=executor,
    )


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


def _chat_provider(settings: MemoPilotSettings) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider.from_deepseek_credentials(
        api_key=settings.chat_api_key.get_secret_value(),
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_retry_limit,
        timeout_seconds=settings.llm_timeout_seconds,
        thinking_enabled=settings.llm_thinking_enabled,
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
    "RuntimeBundle",
    "WorkerBundle",
    "build_app",
    "build_effects",
    "build_runtime_bundle",
    "build_worker",
]
