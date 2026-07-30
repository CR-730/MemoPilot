from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memopilot.bootstrap import AppRuntime, build_app_runtime, build_runtime_bundle
from memopilot.config import MemoPilotSettings
from memopilot.extensions.events import EventBus
from memopilot.extensions.mcp import McpServerConfig
from memopilot.runtime.contracts import ChatMessage, FunctionCall, ModelResponse, ToolSchema
from memopilot.runtime.engine import TurnInput
from memopilot.runtime.tools import ToolRegistry

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


async def test_app_runtime_connects_mcp_before_scheduler_can_start() -> None:
    events: list[str] = []

    class Lifecycle:
        async def start(self) -> None:
            events.append("channel")

    class Registry:
        diagnostics: tuple[str, ...] = ()

        async def load_and_connect_all(self) -> None:
            events.append("mcp")

    app = AppRuntime(
        scheduler=object(),  # type: ignore[arg-type]
        agent_loop=object(),  # type: ignore[arg-type]
        redis=object(),  # type: ignore[arg-type]
        repository=object(),  # type: ignore[arg-type]
        channel=Lifecycle(),  # type: ignore[arg-type]
        console=Lifecycle(),  # type: ignore[arg-type]
        runtime=SimpleNamespace(mcp_registry=Registry()),  # type: ignore[arg-type]
    )

    await app.start()

    assert events == ["channel", "channel", "mcp"]


async def test_app_runtime_registers_feishu_media_senders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_calls: list[tuple[str, str, str, str | None]] = []
    captured: dict[str, object] = {}

    class FakeRedis:
        async def aclose(self) -> None:
            return None

    class FakeChannel:
        async def send(
            self,
            chat_id: str,
            message: str,
            *,
            provider_uuid: str | None = None,
        ) -> None:
            del chat_id, message, provider_uuid

        async def send_image(
            self,
            chat_id: str,
            image: str,
            *,
            provider_uuid: str,
        ) -> None:
            media_calls.append(("image", chat_id, image, provider_uuid))

        async def send_file(
            self,
            chat_id: str,
            file: str,
            *,
            provider_uuid: str,
            name: str | None = None,
        ) -> None:
            media_calls.append(("file", chat_id, file, f"{provider_uuid}:{name}"))

    async def stop_after_capture(*args: object, **kwargs: object) -> object:
        del args
        captured["message_push"] = kwargs["message_push"]
        raise RuntimeError("captured")

    monkeypatch.setattr(
        "memopilot.bootstrap.Redis.from_url",
        lambda *args, **kwargs: FakeRedis(),
    )
    monkeypatch.setattr(
        "memopilot.bootstrap._feishu_channel",
        lambda *args, **kwargs: FakeChannel(),
    )
    monkeypatch.setattr(
        "memopilot.bootstrap.build_runtime_bundle",
        stop_after_capture,
    )
    settings = MemoPilotSettings(
        workspace=tmp_path,
        chat_api_key="chat-key",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_api_key="embedding-key",
        embedding_dimension=2,
        feishu_enabled=True,
        feishu_app_id="app-id",
        feishu_app_secret="app-secret",
        _env_file=None,
    )

    with pytest.raises(RuntimeError, match="captured"):
        await build_app_runtime(settings)

    push = captured["message_push"]
    await push.execute(  # type: ignore[attr-defined]
        channel="feishu",
        chat_id="chat",
        image="meme.png",
        provider_uuid="image-uuid",
    )
    await push.execute(  # type: ignore[attr-defined]
        channel="feishu",
        chat_id="chat",
        file="report.pdf",
        provider_uuid="file-uuid",
    )
    cli_result = await push.execute(  # type: ignore[attr-defined]
        channel="cli",
        chat_id="chat",
        image="meme.png",
    )

    assert [(kind, chat_id, path) for kind, chat_id, path, _ in media_calls] == [
        ("image", "chat", "meme.png"),
        ("file", "chat", "report.pdf"),
    ]
    assert [provider for *_, provider in media_calls] == [
        "image-uuid",
        "file-uuid:report.pdf",
    ]
    assert "不支持发送图片" in cli_result


async def test_runtime_bundle_rolls_back_extensions_when_background_construction_fails(
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
    monkeypatch.setattr("memopilot.bootstrap.TaskDispatcher", fail_executor)
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
            outbound=object(),  # type: ignore[arg-type]
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
    assert {
        "recall_memory",
        "memorize",
        "forget_memory",
    } <= bundle.tools.get_always_on_names()
    assert {"schedule", "list_schedules", "cancel_schedule"} <= tool_names
    assert any("shell_restore" in hook_id for hook_id in bundle.hook_ids)
    assert any("shell_safety" in hook_id for hook_id in bundle.hook_ids)
    assert bundle.runtime is not None
    assert bundle.task_dispatcher is None
    assert bundle.memory_tasks.repository is bundle.repository
    assert settings.operational_database.exists()
    assert settings.memory_database.exists()
    assert "主动推送" in (settings.workspace / "PROACTIVE_CONTEXT.md").read_text(
        encoding="utf-8"
    )
    await bundle.close_extensions()


async def test_runtime_bundle_injects_one_derived_memory_window(
    tmp_path: Path,
) -> None:
    settings = MemoPilotSettings(
        workspace=tmp_path,
        memory_window=41,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_dimension=2,
        _env_file=None,
    )

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        outbound=object(),  # type: ignore[arg-type]
    )
    try:
        assert bundle.task_dispatcher is not None
        assert bundle.task_dispatcher._passive._history_limit == 22
        assert bundle.memory_tasks.consolidation.keep_count == 22
        assert bundle.memory_tasks.consolidation.min_new_messages == 11
        assert bundle.memory_tasks.consolidation.recent_turn_count == 11
    finally:
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
    assert [skill.name for skill in bundle.skills.background_candidates()] == [
        "create-drift-skill",
        "research",
    ]
    assert [(item.plugin_id, item.code) for item in bundle.plugin_diagnostics] == [
        ("broken", "import_failed")
    ]
    assert bundle.skill_diagnostics == ()
    assert bundle.plugin_manager.get_plugin("demo").context.kv_store.get("starts") == 1

    await bundle.runtime.run(TurnInput("feishu:user", "你好", system_prompt="核心"))
    assert "启动即生效的插件提示" in (provider.messages[0][0].content or "")
    assert "# Skill: tool-failure-recovery" in (provider.messages[0][0].content or "")
    await bundle.close_extensions()


async def test_runtime_bundle_builds_core_runner_when_outbound_is_available(
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

    bundle = await build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        outbound=object(),  # type: ignore[arg-type]
    )
    assert "recall_memory" in {
        schema["function"]["name"] for schema in bundle.tools.schemas()
    }
    assert bundle.memory_tasks.repository is bundle.repository
    assert bundle.task_dispatcher is not None
    assert settings.operational_database.exists()
    assert settings.memory_database.exists()
    await bundle.close_extensions()


async def test_runtime_starts_local_mcp_tools_without_blocking_core_runtime(
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
    runner = await build_runtime_bundle(
        settings,
        chat_provider=_ChatProvider(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
    )
    try:
        assert [
            skill.name for skill in runner.skills.background_candidates()
        ] == ["create-drift-skill"]
        runner.mcp_registry.start_connect_all_background()
        for _ in range(100):
            if "mcp_fake__echo" in runner.tools.tool_names:
                break
            await asyncio.sleep(0.02)
        assert "mcp_fake__echo" in runner.tools.tool_names
        assert [
            skill.name for skill in runner.skills.background_candidates()
        ] == ["create-drift-skill", "mcp_research"]
        assert runner.mcp_registry.diagnostics == ()
    finally:
        await runner.close_extensions()
