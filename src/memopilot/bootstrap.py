"""App、Worker、Effect 与分层记忆的集中装配入口。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
from memopilot.extensions.events import EventBus
from memopilot.extensions.mcp_manage_tools import register_mcp_management_tools
from memopilot.extensions.mcp_registry import McpServerRegistry
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.extensions.plugins import PluginDiagnostic
from memopilot.extensions.skills import (
    SkillCatalog,
    SkillDiagnostic,
    SkillLoader,
)
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
from memopilot.proactive.content_turn import AgentTick, AgentTickDeps
from memopilot.proactive.dedupe import MessageDeduper
from memopilot.proactive.drift import DriftSkillSelector
from memopilot.proactive.investigation import ToolContentFetcher
from memopilot.proactive.job_handler import (
    OperationalProactiveJobEnqueuer,
    ProactiveJobHandlerService,
)
from memopilot.proactive.mcp_sources import ProactiveSourceGateway, load_proactive_sources
from memopilot.proactive.service import ProactiveService
from memopilot.proactive.store import ProactiveRepository
from memopilot.runtime.engine import AgentRuntime
from memopilot.runtime.providers import ChatProvider, OpenAICompatibleProvider
from memopilot.runtime.tool_search import build_tool_search_tool
from memopilot.runtime.tools import Tool, ToolRegistry
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.scheduling.job_executor import SystemJobRouter
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.scheduling.runner import SchedulerProcess, SystemScheduler
from memopilot.scheduling.service import ScheduleService
from memopilot.scheduling.tools import build_schedule_tools
from memopilot.tasks.interrupts import RedisInterruptSignal
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import FenceToken, OperationalRepository, RunClaim
from memopilot.tasks.outbox import OutboxDispatcher
from memopilot.tasks.redis_queue import RedisTaskQueue
from memopilot.worker.service import WorkerService

_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "builtin_skills"
_PROACTIVE_CONTEXT_TEMPLATE = """# Proactive Context

在这里写用户当前对主动推送的明确要求和规则。

- 每轮主动唤醒都会读取这份文件，并把它视为必须遵守的规则。
- 适合写白名单、黑名单、过滤条件、优先级和必须先验证的步骤。
- 这里只定义规则，不保存候选资讯或冗长过程。
"""


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
    _extensions_started: bool = False

    @property
    def mcp_diagnostics(self) -> list[str]:
        return list(self.runtime.mcp_registry.diagnostics)

    async def start_extensions(self) -> None:
        if self._extensions_started:
            return
        self._extensions_started = True
        self.runtime.mcp_registry.start_connect_all_background()

    async def close(self) -> None:
        await self.runtime.close_extensions()
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
    service: SchedulerProcess
    redis: Redis

    async def close(self) -> None:
        await self.redis.aclose()


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    repository: OperationalRepository
    tools: ToolRegistry
    memory_engine: LayeredMemoryEngine
    memory_jobs: MemoryJobRouter
    runtime: AgentRuntime
    executor: RuntimeJobExecutor
    skills: SkillCatalog
    event_bus: EventBus
    plugin_manager: PluginManager
    hook_ids: tuple[str, ...]
    mcp_registry: McpServerRegistry
    plugin_diagnostics: tuple[PluginDiagnostic, ...]
    skill_diagnostics: tuple[SkillDiagnostic, ...]

    async def close_extensions(self) -> None:
        """按依赖逆序幂等拆除扩展贡献；半启动状态也可安全调用。"""
        await self.event_bus.drain()
        for hook_id in self.hook_ids:
            self.tools.unregister_hook(hook_id)
        await self.plugin_manager.unload_all()
        await self.mcp_registry.shutdown()
        await self.event_bus.aclose()


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
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    queue = RedisTaskQueue(redis)
    memory_scheduler = MemoryMaintenanceScheduler(
        repository,
        enabled=settings.memory_optimizer_enabled,
        interval=timedelta(seconds=settings.memory_optimizer_interval_seconds),
    )
    schedules = ScheduleService(
        ScheduleRepository(
            settings.operational_database,
            busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
        )
    )
    scheduler = SystemScheduler(
        repository,
        memory_scheduler=memory_scheduler,
        schedule_service=schedules,
        proactive_tick_seconds=settings.proactive_tick_seconds,
        proactive_enabled=settings.proactive_enabled,
    )
    return SchedulerBundle(
        SchedulerProcess(
            scheduler,
            OutboxDispatcher(
                repository,
                queue,
                owner_id=f"scheduler-{uuid4().hex[:8]}",
            ),
        ),
        redis,
    )


async def build_worker(settings: MemoPilotSettings) -> WorkerBundle:
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
    try:
        runtime_bundle = await build_runtime_bundle(
            settings,
            chat_provider=provider,
            final_response_dispatcher=dispatcher,
            repository=repository,
        )
    except BaseException:
        await transport.stop()
        await redis.aclose()
        raise
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
    return WorkerBundle(
        service,
        transport,
        redis,
        runtime_bundle,
    )


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


async def build_runtime_bundle(
    settings: MemoPilotSettings,
    *,
    chat_provider: ChatProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    tools: Iterable[Tool] = (),
    final_response_dispatcher: FinalResponseDispatcher | None = None,
    repository: OperationalRepository | None = None,
    skills: SkillCatalog | None = None,
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
    _ensure_proactive_context(settings.workspace / "PROACTIVE_CONTEXT.md")
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
    registry = ToolRegistry(configured_tools)
    registry.register(build_recall_memory_tool(memory_engine), always_on=True)
    registry.register(build_tool_search_tool(registry), always_on=True)
    schedule_service = ScheduleService(
        ScheduleRepository(
            settings.operational_database,
            busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
        )
    )
    for schedule_tool in build_schedule_tools(schedule_service):
        registry.register(schedule_tool)
    event_bus = EventBus()
    skill_holder: list[SkillCatalog] = []
    procedure_tagger: ChatProcedureTagger | None = None

    def refresh_skill_availability() -> None:
        if skill_holder:
            skill_holder[0].refresh_available_tools(frozenset(registry.tool_names))
        if procedure_tagger is not None:
            procedure_tagger.allowed_tools = {
                name.casefold() for name in registry.tool_names
            }

    mcp_registry = McpServerRegistry(
        settings.workspace / "mcp_servers.json",
        registry,
        on_tools_changed=refresh_skill_availability,
    )
    manager = PluginManager(
        [settings.plugins_dir],
        event_bus=event_bus,
        tool_registry=registry,
        workspace=settings.workspace,
        memory_engine=memory_engine,
    )
    hook_ids: tuple[str, ...] = ()
    try:
        await mcp_registry.import_configs(settings.mcp_servers)
        proactive_source_path = settings.workspace / "proactive_sources.json"
        proactive_sources = load_proactive_sources(proactive_source_path)
        missing_proactive_servers = sorted(
            {source.server for source in proactive_sources}
            - set(mcp_registry.server_ids)
        )
        if missing_proactive_servers:
            raise ValueError(
                "Proactive Source 引用了未配置的 MCP server_id: "
                + ", ".join(missing_proactive_servers)
            )
        register_mcp_management_tools(registry, mcp_registry)
        await manager.load_all()
        registry.register_hooks(manager.tool_hooks)
        hook_ids = tuple(hook.hook_id for hook in manager.tool_hooks)
        if skills is None:
            skill_result = SkillLoader(
                builtin_root=_BUILTIN_SKILLS_DIR,
                workspace_root=settings.skills_dir,
                available_tools=frozenset(registry.tool_names),
            ).load()
            active_skills = SkillCatalog(skill_result.skills)
            skill_diagnostics = skill_result.diagnostics
        else:
            active_skills = skills
            active_skills.refresh_available_tools(frozenset(registry.tool_names))
            skill_diagnostics = ()
        skill_holder.append(active_skills)
        procedure_tagger = ChatProcedureTagger(
            provider,
            allowed_tools={
                *(name for name in registry.tool_names),
                "memorize",
                "forget_memory",
            },
            allowed_skills=(
                {skill.name for skill in skill_result.skills}
                if skills is None
                else set()
            ),
        )
        memorizer = MemoryMemorizer(
            store,
            embedding_provider,
            procedure_tagger=procedure_tagger,
        )
        registry.register(build_memorize_tool(memorizer))
        registry.register(build_forget_memory_tool(store))
        refresh_skill_availability()
        runtime = AgentRuntime(
            provider,
            registry,
            max_iterations=settings.llm_max_iterations,
            memory_engine=memory_engine,
            memory_profile=markdown,
            modules=manager.phase_modules,
            prompt_blocks=manager.prompt_blocks,
            event_bus=event_bus,
            skills=active_skills,
            tool_search_enabled=settings.tool_search_enabled,
            prompt_workspace=settings.workspace,
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
        proactive_repository = ProactiveRepository(settings.proactive_database)
        proactive_gateway = ProactiveSourceGateway(
            proactive_repository,
            config_path=proactive_source_path,
            caller_for_server=mcp_registry.caller,
        )
        proactive_enqueuer = OperationalProactiveJobEnqueuer(operational)

        def build_proactive_service(claim: RunClaim, lease: FenceToken) -> ProactiveService:
            job = operational.get_job(claim.job_id)
            if job is None:
                raise KeyError(claim.job_id)

            def assert_proactive_current() -> None:
                operational.assert_current_fence_and_activity(
                    lease,
                    expected_activity_version=job.activity_version,
                )

            proactive_web_fetch = _proactive_web_fetch_tool(registry)
            proactive_web_search = _proactive_named_tool(registry, "web_search")

            async def search_content(**arguments: object) -> object:
                if proactive_web_search is None:
                    return {"error": "web_search tool not configured"}
                return await proactive_web_search.handler(**arguments)

            agent_tick = AgentTick(
                provider,
                AgentTickDeps(
                    memory=memory_engine,
                    content_fetcher=(
                        ToolContentFetcher(proactive_web_fetch)
                        if proactive_web_fetch is not None
                        else None
                    ),
                    web_search=(
                        search_content if proactive_web_search is not None else None
                    ),
                    recent_chat=lambda session_key, n: _proactive_recent_chat(
                        operational,
                        session_key,
                        now=datetime.now(UTC),
                        limit=n,
                    ),
                ),
                max_steps=20,
                checkpoint=assert_proactive_current,
            )

            return ProactiveService(
                proactive_repository,
                proactive_gateway,
                agent_tick,
                skill_catalog=active_skills,
                job_enqueuer=proactive_enqueuer,
                assert_current=assert_proactive_current,
                drift_min_interval=timedelta(
                    hours=settings.drift_min_interval_hours
                ),
                drift_enabled=settings.drift_enabled,
                message_deduper=MessageDeduper(provider),
                memory_text=lambda: markdown.read("MEMORY.md"),
                proactive_context=lambda: _read_optional_text(
                    settings.workspace / "PROACTIVE_CONTEXT.md"
                ),
                recent_context=lambda _session_key, _now: markdown.read(
                    "RECENT_CONTEXT.md"
                ),
            )

        system_jobs = SystemJobRouter(
            operational,
            runtime,
            dispatcher=final_response_dispatcher,
            proactive_handler=(
                None
                if final_response_dispatcher is None
                else ProactiveJobHandlerService(
                    operational=operational,
                    effects=EffectRepository(operational.database),
                    dispatcher=final_response_dispatcher,
                    service_factory=build_proactive_service,
                )
            ),
            drift_selector=DriftSkillSelector(provider, active_skills),
            drift_workspace=settings.workspace,
            shared_tools=registry,
            connected_mcp_servers=lambda: frozenset(mcp_registry.connected_server_ids),
        )
        executor = RuntimeJobExecutor(
            operational,
            runtime,
            final_response_dispatcher=final_response_dispatcher,
            memory_jobs=memory_jobs,
            system_jobs=system_jobs,
        )
        return RuntimeBundle(
            repository=operational,
            tools=registry,
            memory_engine=memory_engine,
            memory_jobs=memory_jobs,
            runtime=runtime,
            executor=executor,
            skills=active_skills,
            event_bus=event_bus,
            plugin_manager=manager,
            hook_ids=hook_ids,
            mcp_registry=mcp_registry,
            plugin_diagnostics=tuple(manager.diagnostics),
            skill_diagnostics=skill_diagnostics,
        )
    except BaseException:
        for hook_id in hook_ids:
            registry.unregister_hook(hook_id)
        await manager.unload_all()
        await mcp_registry.shutdown()
        await event_bus.aclose()
        raise


def _proactive_web_fetch_tool(registry: ToolRegistry) -> Tool | None:
    return _proactive_named_tool(registry, "web_fetch")


def _proactive_named_tool(registry: ToolRegistry, tool_name: str) -> Tool | None:
    exact = registry.get_tool(tool_name)
    if exact is not None:
        return exact
    candidates = tuple(
        tool
        for name in registry.tool_names
        if (tool := registry.get_tool(name)) is not None
        and tool.source.rpartition("/")[2] == tool_name
    )
    return candidates[0] if len(candidates) == 1 else None


def _read_optional_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _ensure_proactive_context(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PROACTIVE_CONTEXT_TEMPLATE, encoding="utf-8")


def _proactive_recent_session(
    repository: OperationalRepository,
    session_key: str,
    now: datetime,
) -> str:
    lines: list[str] = []
    for message in repository.list_recent_messages(session_key, limit=20, before=now):
        if message.role not in {"user", "assistant"}:
            continue
        try:
            created_at = datetime.fromisoformat(message.created_at.replace("Z", "+00:00"))
            created_at = (
                created_at.replace(tzinfo=UTC)
                if created_at.tzinfo is None
                else created_at.astimezone(UTC)
            )
        except ValueError:
            continue
        lines.append(f"{message.role}: {message.content[:300]}")
    return "\n".join(lines)[:3_000]


def _proactive_recent_chat(
    repository: OperationalRepository,
    session_key: str,
    *,
    now: datetime,
    limit: int,
) -> list[dict[str, str]]:
    return [
        {
            "role": message.role,
            "content": message.content[:1_000],
            "created_at": message.created_at,
        }
        for message in repository.list_recent_messages(
            session_key,
            limit=max(1, min(limit, 100)),
            before=now,
        )
        if message.role in {"user", "assistant"}
    ]


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
    "SchedulerBundle",
    "WorkerBundle",
    "build_app",
    "build_effects",
    "build_runtime_bundle",
    "build_scheduler",
    "build_worker",
]
