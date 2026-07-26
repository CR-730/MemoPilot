from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from memopilot.app.service import AppService
from memopilot.bootstrap import (
    build_app,
    build_effects,
    build_runtime_bundle,
    build_scheduler,
    build_worker,
)
from memopilot.config import MemoPilotSettings
from memopilot.extensions.events import EventBus
from memopilot.extensions.mcp import McpServerConfig
from memopilot.runtime.contracts import ChatMessage, FunctionCall, ModelResponse, ToolSchema
from memopilot.runtime.engine import TurnInput
from memopilot.runtime.tools import ToolRegistry
from memopilot.scheduling.runner import SchedulerProcess, SystemScheduler
from memopilot.worker.service import WorkerService

FAKE_MCP_SERVER = Path(__file__).parents[1] / "fixtures" / "fake_mcp_server.py"


class _ChatProvider:
    async def complete(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ToolSchema, ...],
    ) -> ModelResponse:
        del messages, tools
        return ModelResponse(content="完成")


class _CapturingChatProvider(_ChatProvider):
    def __init__(self) -> None:
        self.messages: list[tuple[ChatMessage, ...]] = []

    async def complete(
        self,
        *,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ToolSchema, ...],
    ) -> ModelResponse:
        del tools
        self.messages.append(messages)
        return ModelResponse(content="完成")


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


async def test_runtime_bundle_rolls_back_extensions_when_executor_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "plugins" / "rollback"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        '''
from memopilot.extensions.decorators import on_before_turn, tool
from memopilot.extensions.plugin_base import Plugin

class RollbackPlugin(Plugin):
    name = "rollback"

    @tool("rollback_tool")
    async def rollback_tool(self, event):
        return "never"

    @on_before_turn()
    async def before_turn(self, event):
        return event

    async def terminate(self):
        (self.context.workspace / "terminated.txt").write_text("yes", encoding="utf-8")
''',
        encoding="utf-8",
    )
    registries: list[ToolRegistry] = []
    buses: list[EventBus] = []

    class CapturingRegistry(ToolRegistry):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            registries.append(self)

    class CapturingBus(EventBus):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            buses.append(self)

    def fail_executor(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("executor failed")

    monkeypatch.setattr("memopilot.bootstrap.ToolRegistry", CapturingRegistry)
    monkeypatch.setattr("memopilot.bootstrap.EventBus", CapturingBus)
    monkeypatch.setattr("memopilot.bootstrap.RuntimeJobExecutor", fail_executor)
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        await build_runtime_bundle(
            settings,
            chat_provider=_ChatProvider(),  # type: ignore[arg-type]
            embedder=_Embedder(),  # type: ignore[arg-type]
        )

    assert (tmp_path / "terminated.txt").read_text(encoding="utf-8") == "yes"
    assert "rollback_tool" not in registries[-1].tool_names
    assert len(buses[-1]._handlers) == 1
    assert buses[-1]._closed is True


async def test_runtime_bundle_connects_memory_to_agent_and_background_jobs(
    tmp_path: Path,
) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    tool_names = {schema["function"]["name"] for schema in bundle.tools.schemas()}
    assert {
        "tool_search",
        "shell",
        "task_output",
        "task_stop",
        "web_search",
        "web_fetch",
        "read_file",
        "list_dir",
        "fetch_messages",
        "search_messages",
        "message_push",
        "write_file",
        "edit_file",
    } <= tool_names
    assert "recall_memory" in tool_names
    assert {"schedule", "list_schedules", "cancel_schedule"} <= tool_names
    assert any("shell_restore" in hook_id for hook_id in bundle.hook_ids)
    assert any("shell_safety" in hook_id for hook_id in bundle.hook_ids)
    assert bundle.runtime is not None
    assert bundle.executor is not None
    assert bundle.executor._system_jobs is not None
    assert bundle.memory_jobs.repository is bundle.repository
    assert settings.operational_database.exists()
    assert settings.memory_database.exists()
    assert "主动推送" in (settings.workspace / "PROACTIVE_CONTEXT.md").read_text(
        encoding="utf-8"
    )
    await bundle.close_extensions()


async def test_proactive_source_can_reference_server_from_manual_mcp_json(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "mcp_servers.json").write_text(
        json.dumps(
            {"servers": {"feeds": {"command": ["python"], "args": ["server.py"]}}}
        ),
        encoding="utf-8",
    )
    (workspace / "proactive_sources.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "news",
                        "server": "feeds",
                        "channel": "content",
                        "get_tool": "fetch_events",
                        "ack_tool": "ack_event",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = MemoPilotSettings(
        workspace=workspace,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    assert bundle.mcp_registry.server_ids == ("feeds",)
    await bundle.close_extensions()


async def test_runtime_rejects_proactive_source_with_truly_missing_mcp_server(
    tmp_path: Path,
) -> None:
    (tmp_path / "proactive_sources.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "news",
                        "server": "missing",
                        "channel": "content",
                        "get_tool": "fetch_events",
                        "ack_tool": "ack_event",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="missing"):
        await build_runtime_bundle(
            settings,
            chat_provider=_ChatProvider(),  # type: ignore[arg-type]
            embedder=_Embedder(),  # type: ignore[arg-type]
        )


async def test_runtime_bundle_wires_plugin_tools_and_hooks(tmp_path: Path) -> None:
    plugin = tmp_path / "plugins" / "rewrite"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        '''
from memopilot.extensions.decorators import on_tool_pre, tool
from memopilot.extensions.plugin_base import Plugin

class Rewrite(Plugin):
    name = "rewrite"

    @tool(name="plugin_echo")
    async def echo(self, event, value: str):
        return value

    @on_tool_pre(tool_name="plugin_echo")
    async def rewrite(self, event):
        return {"value": "rewritten"}
''',
        encoding="utf-8",
    )
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    observation = await bundle.tools.execute(
        FunctionCall("call-1", "plugin_echo", {"value": "original"})
    )
    assert observation.result == "rewritten"
    await bundle.close_extensions()
    await bundle.close_extensions()
    assert "plugin_echo" not in bundle.tools.tool_names
    with pytest.raises(RuntimeError, match="关闭"):
        bundle.event_bus.enqueue("before_turn", {})


async def test_runtime_bundle_discovers_workspace_plugin_and_skill(tmp_path: Path) -> None:
    plugin = tmp_path / "plugins" / "demo"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        '''
from memopilot.extensions.decorators import tool
from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.prompts import PromptBlock

class Demo(Plugin):
    name = "demo"

    async def initialize(self):
        self.context.kv_store.increment("starts")

    def prompt_render_modules(self):
        return [PromptBlock("demo.identity", "启动即生效的插件提示", priority=10)]

    @tool(name="search")
    async def search(self, event, query: str):
        return query
''',
        encoding="utf-8",
    )
    broken = tmp_path / "plugins" / "broken"
    broken.mkdir()
    (broken / "plugin.py").write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    skill = tmp_path / "skills" / "research"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: research
description: 调研
background_allowed: true
required_tools: [search]
---
先检索再总结。
""",
        encoding="utf-8",
    )
    settings = MemoPilotSettings(
        workspace=tmp_path,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    provider = _CapturingChatProvider()
    bundle = await build_runtime_bundle(
        settings,
        chat_provider=provider,  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
    )

    assert "search" in bundle.tools.tool_names
    assert [skill.name for skill in bundle.skills.background_candidates()] == ["research"]
    assert [(item.plugin_id, item.code) for item in bundle.plugin_diagnostics] == [
        ("broken", "import_failed")
    ]
    assert bundle.skill_diagnostics == ()
    assert bundle.plugin_manager.get_plugin("demo").context.kv_store.get("starts") == 1

    await bundle.runtime.run(TurnInput("feishu:user", "你好", system_prompt="核心"))
    assert "启动即生效的插件提示" in (provider.messages[0][0].content or "")
    assert "# Skill: tool-failure-recovery" in (provider.messages[0][0].content or "")
    await bundle.close_extensions()


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
    worker = await build_worker(settings)

    assert isinstance(app.service, AppService)
    assert isinstance(worker.service, WorkerService)
    assert "recall_memory" in {
        schema["function"]["name"] for schema in worker.runtime.tools.schemas()
    }
    assert worker.runtime.memory_jobs.repository is worker.runtime.repository
    assert worker.runtime.executor._system_jobs is not None
    assert worker.runtime.executor._system_jobs.proactive_handler is not None
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


async def test_scheduler_builds_system_tick_and_outbox_process_without_model_credentials(
    tmp_path: Path,
) -> None:
    settings = MemoPilotSettings(workspace=tmp_path / "workspace", _env_file=None)

    bundle = build_scheduler(settings)

    assert isinstance(bundle.service, SchedulerProcess)
    assert isinstance(bundle.service.scheduler, SystemScheduler)
    assert bundle.service.scheduler.proactive_tick_seconds == 300
    await bundle.close()


async def test_worker_starts_local_mcp_tools_without_blocking_core_runtime(
    tmp_path: Path,
) -> None:
    skill = tmp_path / "workspace" / "skills" / "mcp_research"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: mcp_research
description: MCP 调研
background_allowed: true
required_tools: [mcp_fake__echo]
---
调用 MCP 工具调研。
""",
        encoding="utf-8",
    )
    settings = MemoPilotSettings(
        workspace=tmp_path / "workspace",
        chat_api_key="chat-secret",
        embedding_api_key="embedding-secret",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        feishu_app_id="cli-app",
        feishu_app_secret="feishu-secret",
        mcp_servers=(
            McpServerConfig(
                server_id="fake",
                command=(sys.executable,),
                args=(str(FAKE_MCP_SERVER),),
                startup_timeout_seconds=5,
                call_timeout_seconds=1,
            ),
        ),
        _env_file=None,
    )
    worker = await build_worker(settings)
    try:
        assert worker.runtime.skills.background_candidates() == ()
        await worker.start_extensions()
        await worker.start_extensions()
        for _ in range(100):
            if "mcp_fake__echo" in worker.runtime.tools.tool_names:
                break
            await asyncio.sleep(0.02)
        assert "mcp_fake__echo" in worker.runtime.tools.tool_names
        assert [
            skill.name for skill in worker.runtime.skills.background_candidates()
        ] == ["mcp_research"]
        assert worker.mcp_diagnostics == []
    finally:
        await worker.close()
