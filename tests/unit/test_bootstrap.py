from __future__ import annotations

from pathlib import Path

from memopilot.app.service import AppService
from memopilot.bootstrap import build_app, build_effects, build_runtime_bundle, build_worker
from memopilot.config import MemoPilotSettings
from memopilot.extensions.hooks import ToolHook, ToolHookDecision
from memopilot.extensions.plugins import ExtensionRegistry
from memopilot.runtime.contracts import ChatMessage, FunctionCall, ModelResponse, ToolSchema
from memopilot.runtime.tools import Tool
from memopilot.worker.service import WorkerService


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
        _env_file=None,
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


async def test_runtime_bundle_wires_plugin_tools_and_hooks(tmp_path: Path) -> None:
    async def echo(value: str) -> str:
        return value

    async def rewrite(tool_name: str, arguments: dict[str, object]) -> ToolHookDecision:
        del tool_name, arguments
        return ToolHookDecision(arguments={"value": "rewritten"})

    extensions = ExtensionRegistry(
        tools=(
            Tool(
                "plugin_echo",
                "echo",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                echo,
            ),
        ),
        tool_hooks=(ToolHook("rewrite", before=rewrite),),
    )
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        extensions=extensions,
    )

    observation = await bundle.tools.execute(
        FunctionCall("call-1", "plugin_echo", {"value": "original"})
    )
    assert observation.result == "rewritten"


async def test_builds_separate_app_and_worker_without_starting_scheduler(
    tmp_path: Path,
) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path / "workspace",
        chat_api_key="chat-secret",
        embedding_api_key="embedding-secret",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        feishu_app_id="cli-app",
        feishu_app_secret="feishu-secret",
        _env_file=None,
    )

    app = build_app(settings)
    worker = build_worker(settings)

    assert isinstance(app.service, AppService)
    assert isinstance(worker.service, WorkerService)
    assert "recall_memory" in {
        schema["function"]["name"] for schema in worker.runtime.tools.schemas()
    }
    assert worker.runtime.memory_jobs.repository is worker.runtime.repository
    assert settings.operational_database.exists()
    assert settings.memory_database.exists()
    assert not hasattr(app, "scheduler")
    assert not hasattr(worker, "scheduler")
    await app.close()
    await worker.close()


async def test_effects_process_does_not_require_model_credentials(tmp_path: Path) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path / "workspace",
        feishu_app_id="cli-app",
        feishu_app_secret="feishu-secret",
        _env_file=None,
    )

    effects = build_effects(settings)

    assert effects.service is not None
    await effects.close()
