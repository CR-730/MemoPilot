"""AppRuntime、调度、主动链路与分层记忆的集中装配入口。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from redis.asyncio import Redis

from memopilot.bus.events import InboundMessage
from memopilot.channels.base import AttachmentStore, SessionIdentityIndex
from memopilot.channels.contracts import (
    InboundHandler,
    InterruptAcknowledgement,
)
from memopilot.channels.feishu import FeishuChannel
from memopilot.channels.ipc import IPCServerChannel
from memopilot.config import MemoPilotSettings
from memopilot.delivery.feishu_live import FeishuLiveProgress
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
from memopilot.memory.tasks import MemoryTaskRouter
from memopilot.memory.tools import (
    build_forget_memory_tool,
    build_memorize_tool,
    build_recall_memory_tool,
)
from memopilot.memory.vectorization import VectorizationService
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
from memopilot.proactive.loop import ProactiveLoop
from memopilot.proactive.mcp_sources import ProactiveSourceGateway, load_proactive_sources
from memopilot.proactive.service import ProactiveService
from memopilot.proactive.store import ProactiveRepository
from memopilot.runtime.agent_loop import AgentLoop
from memopilot.runtime.background import CoreRunner
from memopilot.runtime.common_tools import register_common_tools
from memopilot.runtime.common_tools.http import SharedHttpResources
from memopilot.runtime.common_tools.message_push import MessagePushTool
from memopilot.runtime.common_tools.vision import build_read_image_vision_tool
from memopilot.runtime.engine import AgentRuntime
from memopilot.runtime.outbound import OutboundPort, PushToolOutboundPort
from memopilot.runtime.providers import ChatProvider, OpenAICompatibleProvider, VisionProvider
from memopilot.runtime.react import ReActProgressObserver
from memopilot.runtime.tool_search import build_tool_search_tool
from memopilot.runtime.tools import Tool, ToolRegistry
from memopilot.scheduling.drift_executor import DriftExecutor
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.scheduling.scheduler import SchedulerService, SystemScheduler
from memopilot.scheduling.service import ScheduleService
from memopilot.scheduling.tools import build_schedule_tools
from memopilot.tasks.lease import SessionLeaseManager
from memopilot.tasks.operational import FenceToken, OperationalRepository
from memopilot.tasks.redis_queue import RedisTaskQueue
from memopilot.tasks.session_coordination import RedisSessionCoordinator

_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "builtin_skills"
_BUILTIN_PLUGINS_DIR = Path(__file__).resolve().parent / "builtin_plugins"
_PROACTIVE_CONTEXT_TEMPLATE = """# Proactive Context

在这里写用户当前对主动推送的明确要求和规则。

- 每轮主动唤醒都会读取这份文件，并把它视为必须遵守的规则。
- 适合写白名单、黑名单、过滤条件、优先级和必须先验证的步骤。
- 这里只定义规则，不保存候选资讯或冗长过程。
"""

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    repository: OperationalRepository
    tools: ToolRegistry
    memory_engine: LayeredMemoryEngine
    memory_tasks: MemoryTaskRouter
    runtime: AgentRuntime
    core_runner: CoreRunner | None
    skills: SkillCatalog
    event_bus: EventBus
    plugin_manager: PluginManager
    hook_ids: tuple[str, ...]
    mcp_registry: McpServerRegistry
    http_resources: SharedHttpResources
    plugin_diagnostics: tuple[PluginDiagnostic, ...]
    skill_diagnostics: tuple[SkillDiagnostic, ...]

    async def close_extensions(self) -> None:
        """按依赖逆序幂等拆除扩展贡献；半启动状态也可安全调用。"""
        await self.event_bus.drain()
        for hook_id in self.hook_ids:
            self.tools.unregister_hook(hook_id)
        await self.plugin_manager.unload_all()
        await self.mcp_registry.shutdown()
        await self.http_resources.aclose()
        await self.event_bus.aclose()


@dataclass(slots=True)
class AppRuntime:
    """原型式单进程运行时；Redis 只承担后台调度、抢占和恢复。"""

    scheduler: SchedulerService
    agent_loop: AgentLoop
    redis: Redis
    repository: OperationalRepository
    channel: FeishuChannel
    console: IPCServerChannel
    runtime: RuntimeBundle
    _started: bool = False

    @property
    def mcp_diagnostics(self) -> tuple[str, ...]:
        return tuple(self.runtime.mcp_registry.diagnostics)

    async def start(self) -> None:
        if self._started:
            return
        await self.console.start()
        await self.channel.start()
        await self.runtime.mcp_registry.load_and_connect_all()
        self._started = True

    async def run_forever(self) -> None:
        tasks = (
            asyncio.create_task(self.agent_loop.run_forever(), name="memopilot-agent-loop"),
            asyncio.create_task(self.scheduler.run_forever(), name="memopilot-scheduler"),
        )
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        self.agent_loop.stop()
        await self.channel.stop()
        await self.console.stop()
        await self.runtime.close_extensions()
        await self.redis.aclose()
        self._started = False


async def build_runtime_bundle(
    settings: MemoPilotSettings,
    *,
    chat_provider: ChatProvider | None = None,
    fast_provider: ChatProvider | None = None,
    vl_provider: VisionProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    tools: Iterable[Tool] = (),
    outbound: OutboundPort | None = None,
    message_push: MessagePushTool | None = None,
    progress_factory: Callable[[InboundMessage], ReActProgressObserver | None] | None = None,
    repository: OperationalRepository | None = None,
    skills: SkillCatalog | None = None,
) -> RuntimeBundle:
    """从类型化配置创建 Agent 与分层记忆链路。"""
    configured_tools = tuple(tools)
    migrate_all_databases(settings)
    operational = repository or OperationalRepository(
        settings.operational_database,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )
    provider = chat_provider or _chat_provider(settings)
    light_provider = fast_provider or _fast_provider(settings) or provider
    vision_provider = vl_provider or _vl_provider(settings)
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
    event_bus = EventBus()
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
        hypothesis_provider=ChatHypothesisProvider(light_provider),
        event_bus=event_bus,
    )
    registry = ToolRegistry(configured_tools)
    http_resources = SharedHttpResources()
    registry.register(build_recall_memory_tool(memory_engine), always_on=True)
    registry.register(build_tool_search_tool(registry), always_on=True)
    register_common_tools(
        registry,
        workspace=settings.workspace,
        repository=operational,
        http_requester=http_resources.external_default,
        push_tool=message_push,
        multimodal=settings.chat_multimodal,
        vl_available=vision_provider is not None,
    )
    if not settings.chat_multimodal and vision_provider is not None and settings.vl_model:
        registry.register(
            build_read_image_vision_tool(vision_provider, workspace=settings.workspace),
            always_on=True,
            risk="read-only",
            search_hint="看图 识图 图片内容 视觉识别 VL",
        )
    schedule_service = ScheduleService(
        ScheduleRepository(
            settings.operational_database,
            busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
        )
    )
    for schedule_tool in build_schedule_tools(schedule_service):
        registry.register(schedule_tool)
    skill_holder: list[SkillCatalog] = []
    procedure_tagger: ChatProcedureTagger | None = None

    def refresh_skill_availability() -> None:
        if skill_holder:
            skill_holder[0].refresh_available_tools(frozenset(registry.tool_names))
        if procedure_tagger is not None:
            procedure_tagger.allowed_tools = {name.casefold() for name in registry.tool_names}

    mcp_registry = McpServerRegistry(
        settings.workspace / "mcp_servers.json",
        registry,
        on_tools_changed=refresh_skill_availability,
    )
    manager = PluginManager(
        [settings.plugins_dir, _BUILTIN_PLUGINS_DIR],
        event_bus=event_bus,
        tool_registry=registry,
        workspace=settings.workspace,
        session_manager=operational,
        memory_engine=memory_engine,
    )
    hook_ids: tuple[str, ...] = ()
    try:
        await mcp_registry.import_configs(settings.mcp_servers)
        proactive_source_path = settings.workspace / "proactive_sources.json"
        proactive_sources = load_proactive_sources(proactive_source_path)
        missing_proactive_servers = sorted(
            {source.server for source in proactive_sources} - set(mcp_registry.server_ids)
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
            light_provider,
            allowed_tools={
                *(name for name in registry.tool_names),
                "memorize",
                "forget_memory",
            },
            allowed_skills=(
                {skill.name for skill in skill_result.skills} if skills is None else set()
            ),
        )
        memorizer = MemoryMemorizer(
            store,
            embedding_provider,
            procedure_tagger=procedure_tagger,
            event_bus=event_bus,
        )
        registry.register(build_memorize_tool(memorizer), always_on=True)
        registry.register(build_forget_memory_tool(store), always_on=True)
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
            context_window_tokens=settings.llm_context_window_tokens,
        )
        memory_tasks = MemoryTaskRouter(
            ConsolidationService(
                settings.operational_database,
                markdown,
                ChatConsolidationExtractor(provider, markdown),
                recent_context=ChatRecentContextCompressor(light_provider),
                keep_count=settings.memory_consolidation_keep_count,
                min_new_messages=settings.memory_consolidation_min_new_messages,
                recent_turn_count=settings.memory_recent_turn_count,
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
                    ChatPostResponseModel(light_provider),
                ),
            ),
        )
        proactive_repository = ProactiveRepository(settings.proactive_database)
        proactive_gateway = ProactiveSourceGateway(
            proactive_repository,
            config_path=proactive_source_path,
            caller_for_server=mcp_registry.caller,
        )

        def build_proactive_service(
            session_key: str,
            activity_version: int,
            lease: FenceToken,
        ) -> ProactiveService:
            def assert_proactive_current() -> None:
                operational.assert_current_fence_and_activity(
                    lease,
                    expected_activity_version=activity_version,
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
                    web_search=(search_content if proactive_web_search is not None else None),
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
                assert_current=assert_proactive_current,
                drift_min_interval=timedelta(hours=settings.drift_min_interval_hours),
                drift_enabled=settings.drift_enabled,
                context_probability=settings.proactive_context_probability,
                active_timezone=ZoneInfo(settings.display_timezone),
                active_start_hour=settings.proactive_active_start_hour,
                active_end_hour=settings.proactive_active_end_hour,
                message_deduper=MessageDeduper(provider),
                memory_text=lambda: markdown.read("MEMORY.md"),
                proactive_context=lambda: _read_optional_text(
                    settings.workspace / "PROACTIVE_CONTEXT.md"
                ),
                recent_context=lambda _session_key, _now: markdown.read("RECENT_CONTEXT.md"),
            )

        proactive_loop = (
            None
            if outbound is None
            else ProactiveLoop(
                outbound=outbound,
                service_factory=build_proactive_service,
            )
        )
        system_jobs = DriftExecutor(
            operational,
            runtime,
            outbound=outbound,
            drift_selector=DriftSkillSelector(provider, active_skills),
            drift_workspace=settings.workspace,
            drift_builtin_skills=_BUILTIN_SKILLS_DIR,
            drift_repository=proactive_repository,
            shared_tools=registry,
            connected_mcp_servers=lambda: frozenset(mcp_registry.connected_server_ids),
        )
        core_runner = (
            CoreRunner(
                runtime,
                repository=operational,
                outbound=outbound,
                memory_tasks=memory_tasks,
                proactive=proactive_loop,
                drift=system_jobs,
                event_bus=event_bus,
                short_term_message_limit=settings.memory_history_limit,
                progress_factory=progress_factory,
            )
            if outbound is not None and proactive_loop is not None
            else None
        )
        return RuntimeBundle(
            repository=operational,
            tools=registry,
            memory_engine=memory_engine,
            memory_tasks=memory_tasks,
            runtime=runtime,
            core_runner=core_runner,
            skills=active_skills,
            event_bus=event_bus,
            plugin_manager=manager,
            hook_ids=hook_ids,
            mcp_registry=mcp_registry,
            http_resources=http_resources,
            plugin_diagnostics=tuple(manager.diagnostics),
            skill_diagnostics=skill_diagnostics,
        )
    except BaseException:
        for hook_id in hook_ids:
            registry.unregister_hook(hook_id)
        await manager.unload_all()
        await mcp_registry.shutdown()
        await http_resources.aclose()
        await event_bus.aclose()
        raise


async def build_app_runtime(settings: MemoPilotSettings) -> AppRuntime:
    settings.validate_runtime_ready()
    settings.validate_app_ready()
    repository = _repository(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    queue = RedisTaskQueue(redis)
    coordinator = RedisSessionCoordinator(redis)

    async def handle_inbound(message: InboundMessage) -> object:
        if message.content.strip() == "/stop":
            await coordinator.request_background_stop(
                message.session_key,
                reason="user_stop",
            )
            return InterruptAcknowledgement(
                message="已收到停止请求。",
                provider_uuid=str(
                    uuid5(
                        NAMESPACE_URL,
                        "memopilot:interrupt:"
                        f"{message.metadata.get('message_id') or message.session_key}",
                    )
                ),
            )
        repository.record_inbound_activity(message)
        return await queue.publish_inbound(
            message,
            stop_key=coordinator.stop_key(message.session_key),
        )

    console = IPCServerChannel(handle_inbound)
    channel = _feishu_channel(settings, repository, inbound_handler=handle_inbound)
    message_push = MessagePushTool()

    async def push_feishu(
        chat_id: str,
        message: str,
        *,
        provider_uuid: str | None = None,
    ) -> None:
        await channel.send(chat_id, message, provider_uuid=provider_uuid)

    async def push_console(
        chat_id: str,
        message: str,
        *,
        provider_uuid: str | None = None,
    ) -> None:
        await console.send(chat_id, message, provider_uuid=provider_uuid)

    async def push_feishu_image(
        chat_id: str,
        image: str,
        *,
        provider_uuid: str,
    ) -> None:
        await channel.send_image(
            chat_id,
            image,
            provider_uuid=provider_uuid,
        )

    async def push_feishu_file(
        chat_id: str,
        file: str,
        name: str | None = None,
        *,
        provider_uuid: str,
    ) -> None:
        await channel.send_file(
            chat_id,
            file,
            provider_uuid=provider_uuid,
            name=name,
        )

    message_push.register_channel(
        "feishu",
        text=push_feishu,
        image=push_feishu_image,
        file=push_feishu_file,
    )
    message_push.register_channel("cli", text=push_console)
    outbound = PushToolOutboundPort(message_push)

    def progress_factory(message: InboundMessage) -> ReActProgressObserver | None:
        if message.channel != "feishu":
            return None
        message_id = str(message.metadata.get("message_id") or message.session_key)
        return FeishuLiveProgress(
            channel,
            chat_id=message.chat_id,
            provider_uuid=str(uuid5(NAMESPACE_URL, f"feishu:{message_id}:live")),
        )

    try:
        runtime = await build_runtime_bundle(
            settings,
            chat_provider=_chat_provider(settings),
            outbound=outbound,
            message_push=message_push,
            progress_factory=progress_factory,
            repository=repository,
        )
    except BaseException:
        await redis.aclose()
        raise
    if runtime.core_runner is None:
        await runtime.close_extensions()
        await redis.aclose()
        raise RuntimeError("后台任务分派器未初始化")

    leases = SessionLeaseManager(
        redis,
        repository,
        ttl=timedelta(seconds=settings.lease_ttl_seconds),
    )
    agent_loop = AgentLoop(
        queue,
        leases,
        runtime.core_runner,
        owner_id=f"agent-{uuid4().hex[:8]}",
        heartbeat_interval=settings.lease_heartbeat_seconds,
        pending_min_idle=timedelta(seconds=settings.reclaim_idle_seconds),
        session_coordinator=coordinator,
    )
    memory_scheduler = MemoryMaintenanceScheduler(
        repository,
        enabled=settings.memory_optimizer_enabled,
        interval=timedelta(seconds=settings.memory_optimizer_interval_seconds),
    )
    schedule_service = ScheduleService(
        ScheduleRepository(
            settings.operational_database,
            busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
        )
    )
    scheduler = SchedulerService(
        SystemScheduler(
            repository,
            memory_scheduler=memory_scheduler,
            schedule_service=schedule_service,
            proactive_tick_seconds=settings.proactive_tick_seconds,
            proactive_enabled=settings.proactive_enabled,
        ),
        queue=queue,
    )
    return AppRuntime(
        scheduler=scheduler,
        agent_loop=agent_loop,
        redis=redis,
        repository=repository,
        channel=channel,
        console=console,
        runtime=runtime,
    )


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
    return OpenAICompatibleProvider.from_routed_credentials(
        api_key=settings.chat_api_key.get_secret_value(),
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_retry_limit,
        timeout_seconds=settings.llm_timeout_seconds,
        thinking_enabled=settings.llm_thinking_enabled,
    )


def _fast_provider(settings: MemoPilotSettings) -> OpenAICompatibleProvider | None:
    if not settings.fast_model or not (
        settings.fast_api_key.get_secret_value() or settings.fast_base_url
    ):
        return None
    base_url = settings.fast_base_url or settings.chat_base_url
    if "googleapis.com" in base_url or "generativelanguage" in base_url:
        return OpenAICompatibleProvider.from_credentials(
            api_key=settings.fast_api_key.get_secret_value()
            or settings.chat_api_key.get_secret_value(),
            base_url=base_url,
            model=settings.fast_model,
            max_output_tokens=settings.llm_max_output_tokens,
            max_retries=settings.llm_retry_limit,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return OpenAICompatibleProvider.from_routed_credentials(
        api_key=settings.fast_api_key.get_secret_value()
        or settings.chat_api_key.get_secret_value(),
        base_url=base_url,
        model=settings.fast_model,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_retry_limit,
        timeout_seconds=settings.llm_timeout_seconds,
        thinking_enabled=False,
    )


def _vl_provider(settings: MemoPilotSettings) -> OpenAICompatibleProvider | None:
    if settings.chat_multimodal or not settings.vl_model:
        return None
    return OpenAICompatibleProvider.from_credentials(
        api_key=settings.vl_api_key.get_secret_value() or settings.chat_api_key.get_secret_value(),
        base_url=settings.vl_base_url or settings.chat_base_url,
        model=settings.vl_model,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_retry_limit,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def _feishu_channel(
    settings: MemoPilotSettings,
    repository: OperationalRepository,
    *,
    inbound_handler: InboundHandler,
) -> FeishuChannel:
    return FeishuChannel(
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret.get_secret_value(),
        inbound_handler=inbound_handler,
        identity_index=SessionIdentityIndex(
            repository,
            channel=settings.feishu_channel_name,
        ),
        attachment_store=AttachmentStore(settings.uploads_dir),
        allow_from=settings.feishu_allow_from,
        channel_name=settings.feishu_channel_name,
    )


__all__ = [
    "AppRuntime",
    "RuntimeBundle",
    "build_app_runtime",
    "build_runtime_bundle",
]
