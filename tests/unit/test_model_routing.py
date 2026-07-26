from __future__ import annotations

from pathlib import Path

from memopilot.bootstrap import build_runtime_bundle
from memopilot.config import MemoPilotSettings
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema


class _Provider:
    async def complete(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ToolSchema, ...],
    ) -> ModelResponse:
        del messages, tools
        return ModelResponse(content="完成")


class _VisionProvider(_Provider):
    async def complete_vision(self, *, data_uri: str, prompt: str) -> str:
        del data_uri, prompt
        return "图片内容"


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


async def test_runtime_routes_lightweight_memory_tasks_to_fast_provider(
    tmp_path: Path,
) -> None:
    main = _Provider()
    fast = _Provider()
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=main,
        fast_provider=fast,
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    assert bundle.runtime._provider is main
    assert bundle.memory_engine.hypothesis_provider.provider is fast
    assert bundle.memory_jobs.consolidation.extractor.provider is main
    assert bundle.memory_jobs.consolidation.recent_context.provider is fast
    assert bundle.memory_jobs.vectorization.implicit_extractor.provider is main
    assert bundle.memory_jobs.vectorization.memorizer.procedure_tagger.provider is fast
    assert bundle.memory_jobs.post_response.worker.model.provider is fast
    await bundle.close_extensions()


async def test_runtime_falls_back_to_main_when_fast_provider_is_missing(
    tmp_path: Path,
) -> None:
    main = _Provider()
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=main,
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    assert bundle.memory_engine.hypothesis_provider.provider is main
    assert bundle.memory_jobs.consolidation.recent_context.provider is main
    assert bundle.memory_jobs.post_response.worker.model.provider is main
    await bundle.close_extensions()


async def test_runtime_registers_vl_tool_only_for_non_multimodal_main(
    tmp_path: Path,
) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path,
        chat_multimodal=False,
        vl_model="vision-model",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=_Provider(),
        vl_provider=_VisionProvider(),
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    assert "read_image_vision" in bundle.tools.tool_names
    await bundle.close_extensions()


async def test_runtime_does_not_register_vl_tool_for_multimodal_main(
    tmp_path: Path,
) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path,
        chat_multimodal=True,
        vl_model="vision-model",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=_Provider(),
        vl_provider=_VisionProvider(),
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    assert "read_image_vision" not in bundle.tools.tool_names
    await bundle.close_extensions()
