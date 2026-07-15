from __future__ import annotations

from pathlib import Path

from memopilot.bootstrap import build_runtime_bundle
from memopilot.config import MemoPilotSettings
from memopilot.runtime.contracts import ChatMessage, ModelResponse, ToolSchema


class _ChatProvider:
    async def complete(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ToolSchema, ...],
    ) -> ModelResponse:
        del messages, tools
        return ModelResponse(content="完成")


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def test_runtime_bundle_connects_memory_to_agent_and_background_jobs(tmp_path: Path) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
    )

    bundle = build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    tool_names = {schema["function"]["name"] for schema in bundle.tools.schemas()}
    assert "recall_memory" in tool_names
    assert bundle.runtime is not None
    assert bundle.executor is not None
    assert bundle.memory_jobs.repository is bundle.repository
    assert settings.operational_database.exists()
    assert settings.memory_database.exists()
