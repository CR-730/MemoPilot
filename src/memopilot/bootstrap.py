"""生产运行时的集中装配入口。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from memopilot.config import MemoPilotSettings
from memopilot.delivery.feishu import FinalResponseDispatcher
from memopilot.memory.consolidation import ConsolidationService
from memopilot.memory.contracts import EmbeddingProvider
from memopilot.memory.engine import LayeredMemoryEngine
from memopilot.memory.markdown import MarkdownMemoryStore
from memopilot.memory.optimizer import MemoryOptimizer
from memopilot.memory.providers import (
    ChatConsolidationExtractor,
    ChatHypothesisProvider,
    ChatOptimizerModel,
    OpenAIEmbeddingProvider,
)
from memopilot.memory.retrieval import MemoryRetriever, RetrievalStore
from memopilot.memory.store import MemoryStore
from memopilot.memory.tools import build_recall_memory_tool
from memopilot.memory.vectorization import VectorizationService
from memopilot.memory.worker import MemoryJobRouter
from memopilot.persistence.memory_metadata import EmbeddingIdentity, ensure_embedding_identity
from memopilot.persistence.migrations import migrate_all_databases
from memopilot.runtime.engine import AgentRuntime
from memopilot.runtime.providers import ChatProvider, OpenAICompatibleProvider
from memopilot.runtime.tools import Tool, ToolRegistry
from memopilot.runtime.worker import RuntimeJobExecutor
from memopilot.tasks.operational import OperationalRepository


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    repository: OperationalRepository
    tools: ToolRegistry
    memory_engine: LayeredMemoryEngine
    memory_jobs: MemoryJobRouter
    runtime: AgentRuntime
    executor: RuntimeJobExecutor


def build_runtime_bundle(
    settings: MemoPilotSettings,
    *,
    chat_provider: ChatProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    tools: Iterable[Tool] = (),
    final_response_dispatcher: FinalResponseDispatcher | None = None,
) -> RuntimeBundle:
    """从类型化配置创建 Worker 使用的完整 Agent 与记忆链路。"""
    migrate_all_databases(settings)
    repository = OperationalRepository(
        settings.operational_database,
        busy_timeout_seconds=settings.sqlite_busy_timeout_seconds,
    )
    provider = chat_provider or OpenAICompatibleProvider.from_deepseek_credentials(
        api_key=settings.chat_api_key.get_secret_value(),
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_retry_limit,
        timeout_seconds=settings.llm_timeout_seconds,
        thinking_enabled=settings.llm_thinking_enabled,
    )
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
    )
    retriever = MemoryRetriever(
        cast(RetrievalStore, store),
        embedding_provider,
        score_threshold=settings.memory_score_threshold,
        relative_delta=settings.memory_relative_delta,
        inject_max_chars=settings.memory_inject_max_chars,
    )
    memory_engine = LayeredMemoryEngine(
        retriever,
        hypothesis_provider=ChatHypothesisProvider(provider),
    )
    registry = ToolRegistry(tools)
    registry.register(build_recall_memory_tool(memory_engine))
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
            ChatConsolidationExtractor(provider),
            keep_count=settings.memory_consolidation_keep_count,
            min_new_messages=settings.memory_consolidation_min_new_messages,
        ),
        VectorizationService(
            settings.operational_database,
            store,
            embedding_provider,
        ),
        MemoryOptimizer(markdown, ChatOptimizerModel(provider)),
        repository,
    )
    executor = RuntimeJobExecutor(
        repository,
        runtime,
        final_response_dispatcher=final_response_dispatcher,
        memory_jobs=memory_jobs,
    )
    return RuntimeBundle(
        repository=repository,
        tools=registry,
        memory_engine=memory_engine,
        memory_jobs=memory_jobs,
        runtime=runtime,
        executor=executor,
    )


__all__ = ["RuntimeBundle", "build_runtime_bundle"]
