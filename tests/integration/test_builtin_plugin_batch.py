from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memopilot.bus.events import InboundMessage
from memopilot.extensions.events import EventBus
from memopilot.extensions.plugin_manager import PluginManager
from memopilot.persistence.migrations import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
from memopilot.runtime.agent_loop import AgentLoop
from memopilot.runtime.background import CoreRunner
from memopilot.runtime.common_tools.message_push import MessagePushTool
from memopilot.runtime.contracts import (
    ChatMessage,
    FunctionCall,
    ModelResponse,
    ToolSchema,
)
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.outbound import PushToolOutboundPort
from memopilot.runtime.providers import ChatProvider
from memopilot.runtime.tools import Tool, ToolRegistry
from memopilot.scheduling.scheduler import ApplicationScheduler
from memopilot.tasks.lease import SessionLease
from memopilot.tasks.operational import OperationalRepository
from memopilot.tasks.redis_queue import QueueMessage

_BUILTIN_ROOT = Path(__file__).parents[2] / "src" / "memopilot" / "builtin_plugins"
_NOW = datetime(2026, 7, 28, tzinfo=UTC)
_LEASE = SessionLease("cli:chat", "agent-1", 1, "lease", "value")


class _Provider(ChatProvider):
    def __init__(self, responses: Sequence[ModelResponse] = ()) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[ChatMessage, ...]] = []

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema],
    ) -> ModelResponse:
        del tools
        self.calls.append(tuple(messages))
        return self.responses.pop(0)


class _Unused:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"不应访问 {name}")


class _Queue:
    def __init__(self, message: QueueMessage) -> None:
        self.message: QueueMessage | None = message
        self.acked = False
        self.published: list[object] = []

    async def ensure_consumer_groups(self) -> None:
        return None

    async def read_next(self, *, consumer_id: str) -> QueueMessage | None:
        del consumer_id
        message, self.message = self.message, None
        return message

    async def acknowledge(self, message: object) -> None:
        del message
        self.acked = True

    async def publish_task_once(self, task: object) -> str:
        self.published.append(task)
        return "2-0"

    async def pending_entries(self, **kwargs: object) -> list[object]:
        del kwargs
        return []


class _Leases:
    ttl_ms = 30_000

    async def acquire(
        self,
        session_key: str,
        *,
        owner_id: str,
        now: datetime,
    ) -> SessionLease:
        del session_key, owner_id, now
        return _LEASE

    async def renew(self, lease: object, *, now: datetime) -> bool:
        del lease, now
        return True

    async def release(self, lease: object) -> bool:
        del lease
        return True

    async def is_absent(self, session_key: str) -> bool:
        del session_key
        return True


class _Coordinator:
    async def clear_background_stop(self, session_key: str) -> None:
        del session_key

    async def stop_reason(self, session_key: str) -> None:
        del session_key
        return None


async def _load(
    plugin_dirs: list[Path],
    *,
    workspace: Path,
    tools: ToolRegistry | None = None,
    repository: OperationalRepository | None = None,
    bus: EventBus | None = None,
) -> PluginManager:
    manager = PluginManager(
        plugin_dirs,
        workspace=workspace,
        tool_registry=tools,
        session_manager=repository,
        event_bus=bus,
    )
    await manager.load_all()
    return manager


def _commit_turn(
    repository: OperationalRepository,
    index: int,
    user: str,
    assistant: str,
) -> None:
    repository.commit_turn(
        InboundMessage(
            "cli",
            "user",
            "chat",
            user,
            timestamp=_NOW,
            metadata={"message_id": f"message-{index}"},
        ),
        assistant_content=assistant,
    )


async def test_setup_helper_short_circuits_with_memopilot_configuration(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    manager = await _load(
        [_BUILTIN_ROOT / "setup_helper"],
        workspace=tmp_path,
    )

    result = await AgentRuntime(
        provider,
        ToolRegistry(),
        modules=manager.phase_modules,
    ).run(TurnInput("cli:chat", "/chatid@MemoPilot"))

    assert provider.calls == []
    assert result.react.exit_reason == "before_turn_abort"
    assert "channel：`cli`" in result.reply
    assert "chat_id：`chat`" in result.reply
    assert "私聊会话已自动登记" in result.reply
    assert "[proactive]\nenabled = true" in result.reply
    assert "proactive_sources.json" in result.reply
    assert "[proactive.target]" not in result.reply
    await manager.unload_all()


async def test_undo_removes_latest_complete_turn_and_rolls_cursor(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    _commit_turn(repository, 1, "第一问", "第一答")
    _commit_turn(repository, 2, "第二问", "第二答")
    with connect_database(database) as connection:
        connection.execute(
            "UPDATE sessions SET last_consolidated_position = 4 "
            "WHERE session_key = 'cli:chat'"
        )
        connection.commit()
    provider = _Provider()
    manager = await _load(
        [_BUILTIN_ROOT / "plugin_undo"],
        workspace=tmp_path,
        repository=repository,
    )

    result = await AgentRuntime(
        provider,
        ToolRegistry(),
        modules=manager.phase_modules,
    ).run(TurnInput("cli:chat", "/undo"))

    assert provider.calls == []
    assert result.react.exit_reason == "before_turn_abort"
    assert "已撤销上一轮对话" in result.reply
    assert "整理游标：4 → 2" in result.reply
    assert "记忆回滚：未执行" in result.reply
    assert [
        (message.role, message.content)
        for message in repository.list_recent_messages("cli:chat", limit=10)
    ] == [("user", "第一问"), ("assistant", "第一答")]
    assert repository.memory_status("cli:chat")[:2] == (2, 2)
    await manager.unload_all()


async def test_meme_runs_through_agent_loop_and_sends_clean_text_and_image(
    tmp_path: Path,
    caplog: Any,
) -> None:
    meme_root = tmp_path / "memes"
    image = meme_root / "shy" / "001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-a-real-png")
    (meme_root / "manifest.json").write_text(
        json.dumps(
            {
                "categories": {
                    "shy": {
                        "desc": "害羞",
                        "aliases": ["bashful"],
                        "enabled": True,
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    repository = OperationalRepository(database)
    inbound = InboundMessage(
        "cli",
        "user",
        "chat",
        "来个表情",
        timestamp=_NOW,
        metadata={"message_id": "meme-message"},
    )
    repository.record_inbound_activity(inbound)
    assert repository.allocate_fence("cli:chat", owner_id="agent-1", now=_NOW) == 1

    async def echo(*, text: str) -> str:
        return text

    tools = ToolRegistry(
        [
            Tool(
                "echo",
                "回显",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                echo,
            )
        ]
    )
    bus = EventBus()
    manager = await _load(
        [_BUILTIN_ROOT / "meme", _BUILTIN_ROOT / "observe"],
        workspace=tmp_path,
        tools=tools,
        repository=repository,
        bus=bus,
    )
    provider = _Provider(
        (
            ModelResponse(
                None,
                (FunctionCall("call-1", "echo", {"text": "ok"}),),
                "tool_calls",
            ),
            ModelResponse("好的 <meme:bashful>", (), "stop"),
        )
    )
    runtime = AgentRuntime(
        provider,
        tools,
        modules=manager.phase_modules,
        event_bus=bus,
    )
    sent_text: list[str] = []
    sent_images: list[str] = []

    async def send_text(
        chat_id: str,
        message: str,
        *,
        provider_uuid: str | None = None,
    ) -> None:
        del chat_id, provider_uuid
        sent_text.append(message)

    async def send_image(
        chat_id: str,
        path: str,
        *,
        provider_uuid: str,
    ) -> None:
        del chat_id, provider_uuid
        sent_images.append(path)

    push = MessagePushTool()
    push.register_channel("cli", text=send_text, image=send_image)
    runner = CoreRunner(
        runtime,
        repository=repository,
        outbound=PushToolOutboundPort(push),
        memory_tasks=_Unused(),
        proactive=_Unused(),
        drift=_Unused(),
        event_bus=bus,
    )
    payload = {
        "channel": "cli",
        "sender": "user",
        "chat_id": "chat",
        "content": "来个表情",
        "timestamp": _NOW.isoformat(),
        "media": [],
        "metadata": {"message_id": "meme-message"},
    }
    queue_message = QueueMessage(
        "memopilot:tasks:p0",
        "1-0",
        "task-1",
        "passive.turn",
        0,
        "cli:chat",
        json.dumps(payload),
    )
    queue = _Queue(queue_message)
    loop_holder: dict[str, AgentLoop] = {}

    async def stop_after_idle(delay: float) -> None:
        del delay
        loop_holder["loop"].stop()

    loop = AgentLoop(
        queue,
        _Leases(),
        runner,
        owner_id="agent-1",
        session_coordinator=_Coordinator(),
        sleep=stop_after_idle,
    )
    loop_holder["loop"] = loop
    caplog.set_level(logging.INFO)

    assert await loop.run_once() is True
    await loop.run_forever(idle_interval=0.001)

    assert queue.acked is True
    assert sent_text == ["好的"]
    assert sent_images == [str(image)]
    system_prompt = "\n".join(
        str(message.content)
        for call in provider.calls
        for message in call
        if message.role == "system"
    )
    assert "# Memes" in system_prompt
    assert "每条回复最多 1 个 <meme:category>" in system_prompt
    assert "以上规则优先于历史回复模式" in system_prompt
    assert "用户要表情 → 不调用 tool_search，不调用任何工具" in system_prompt
    assert repository.list_recent_messages("cli:chat", limit=2)[-1].content == "好的"
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    handling_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "memopilot.runtime.agent_loop"
        and "处理消息 kind=" in record.getMessage()
    ]
    assert handling_logs == [
        "AgentLoop 处理消息 kind=passive.turn session=cli:chat preview='来个表情'"
    ]
    assert "AgentLoop 启动" in log_text
    assert "AgentLoop 停止" in log_text
    assert "[LLM调用]" in log_text
    assert "[工具执行→]" in log_text
    assert "[工具结果←]" in log_text
    assert "[message_push]" in log_text
    assert "observe write" not in log_text
    assert "来个表情" in log_text

    await manager.unload_all()
    await bus.aclose()


async def test_scheduler_logs_start_and_stop(caplog: Any) -> None:
    class _Scheduler:
        poll_interval_seconds = 1.0

        def clock(self) -> datetime:
            return _NOW

        def tick(self, *, now: datetime) -> object:
            assert now == _NOW
            return SimpleNamespace(
                memory=None,
                schedules=SimpleNamespace(tasks=()),
                proactive=None,
            )

        async def sleep(self, delay: float) -> None:
            assert delay == 1.0
            raise asyncio.CancelledError

    caplog.set_level(logging.INFO)
    service = ApplicationScheduler(_Scheduler(), object())  # type: ignore[arg-type]

    with pytest.raises(asyncio.CancelledError):
        await service.run_forever()

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "ApplicationScheduler started" in log_text
    assert "ApplicationScheduler stopped" in log_text
