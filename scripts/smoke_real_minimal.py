"""使用真实模型、Embedding 和飞书执行最小端到端回归。"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from memopilot.bootstrap import AppRuntime, build_app_runtime
from memopilot.bus.events import InboundMessage
from memopilot.config import load_settings
from memopilot.extensions.mcp import McpServerConfig
from memopilot.persistence.conversation import ConversationRepository, MessageRecord
from memopilot.proactive.mcp_sources import ProactiveEvent, ProactiveFetchResult
from memopilot.proactive.store import ProactiveRepository
from memopilot.tasks.redis_queue import RedisTaskQueue

ROOT = Path(__file__).resolve().parents[1]
FAKE_MCP_SERVER = ROOT / "tests" / "fixtures" / "fake_mcp_server.py"
REDIS_URL = "redis://127.0.0.1:6384/15"


def _scalar(database: Path, query: str) -> Any:
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(query).fetchone()
        return None if row is None else row[0]
    finally:
        connection.close()


def _tool_names(message: MessageRecord) -> set[str]:
    names: set[str] = set()
    for group in message.tool_chain:
        calls = group.get("calls")
        if not isinstance(calls, list):
            continue
        names.update(
            str(call.get("name"))
            for call in calls
            if isinstance(call, dict) and call.get("name")
        )
    return names


async def _turn(
    app: AppRuntime,
    queue: RedisTaskQueue,
    *,
    chat_id: str,
    prefix: str,
    content: str,
) -> tuple[InboundMessage, MessageRecord]:
    message = InboundMessage(
        "feishu",
        "real-regression",
        chat_id,
        content,
        timestamp=datetime.now(UTC),
        metadata={"message_id": f"real-{prefix}-{uuid4().hex}"},
    )
    app.repository.record_inbound_activity(message)
    if await queue.publish_inbound(message) is None:
        raise AssertionError("真实入站消息未发布")
    if not await app.agent_loop.run_once():
        raise AssertionError("AgentLoop 未消费真实入站消息")
    history = app.repository.list_recent_messages(message.session_key, limit=2)
    if len(history) != 2 or history[-1].role != "assistant":
        raise AssertionError("真实回复未写入 SQLite")
    return message, history[-1]


async def main() -> None:
    base = load_settings(ROOT / "config.toml")
    target = ConversationRepository(base.operational_database).get_single_private_session()
    if target is None or target.channel != "feishu":
        raise RuntimeError("没有可用的真实飞书私聊目标")

    with tempfile.TemporaryDirectory(prefix="memopilot-real-regression-") as temp:
        settings = load_settings(ROOT / "config.toml", workspace=Path(temp)).model_copy(
            update={
                "redis_url": REDIS_URL,
                "llm_max_output_tokens": 256,
                "llm_max_iterations": 4,
                "llm_thinking_enabled": False,
                "memory_optimizer_enabled": False,
                "drift_enabled": False,
                "mcp_servers": (
                    McpServerConfig(
                        server_id="fake",
                        command=(sys.executable,),
                        args=(str(FAKE_MCP_SERVER),),
                        startup_timeout_seconds=5,
                        call_timeout_seconds=2,
                    ),
                ),
            }
        )
        app = await build_app_runtime(settings)
        queue = RedisTaskQueue(app.redis)
        await app.redis.flushdb()
        await queue.ensure_consumer_groups()
        await app.runtime.mcp_registry.load_and_connect_all()
        prefix = uuid4().hex[:8]
        try:
            _, mcp_reply = await _turn(
                app,
                queue,
                chat_id=target.chat_id,
                prefix=prefix,
                content=(
                    "$tool-failure-recovery 请调用 mcp_fake__echo，参数 text=ECHO-OK；"
                    "根据工具结果只回复 ECHO-OK。"
                ),
            )
            if "mcp_fake__echo" not in _tool_names(mcp_reply):
                raise AssertionError("真实 Runtime 未调用 MCP echo")
            if "ECHO-OK" not in mcp_reply.content:
                raise AssertionError("MCP 工具结果未进入最终回复")
            print("PASS 被动回复 + ReAct + MCP + Skill + 飞书")

            if not await app.agent_loop.run_once() or not await app.agent_loop.run_once():
                raise AssertionError("回复后记忆任务未执行")
            print("PASS 回复后记忆后台任务")

            before = int(_scalar(settings.memory_database, "SELECT COUNT(*) FROM memory_items"))
            _, memory_reply = await _turn(
                app,
                queue,
                chat_id=target.chat_id,
                prefix=prefix,
                content=(
                    f"请调用 memorize 明确记住：我的真实回归代号是蓝鲸-{prefix}；"
                    "只回复“已记住”。"
                ),
            )
            after = int(_scalar(settings.memory_database, "SELECT COUNT(*) FROM memory_items"))
            if after <= before or "memorize" not in _tool_names(memory_reply):
                raise AssertionError("真实记忆没有写入向量库")
            _, recall_reply = await _turn(
                app,
                queue,
                chat_id=target.chat_id,
                prefix=prefix,
                content="我的真实回归代号是什么？只用代号回答。",
            )
            if f"蓝鲸-{prefix}" not in recall_reply.content:
                raise AssertionError("真实自动预检索没有召回刚写入的记忆")
            print("PASS Embedding + 记忆写入/召回 + 飞书")

            schedule_message, _ = await _turn(
                app,
                queue,
                chat_id=target.chat_id,
                prefix=prefix,
                content=(
                    "请调用 schedule：schedule_kind=after，when=5s，"
                    "execution_mode=instant，"
                    f"message=【真实回归】定时链路 {prefix}，name=real-{prefix}；"
                    "只回复“已设置”。"
                ),
            )
            schedule = next(
                item
                for item in app.runtime.scheduler_service.list(schedule_message.session_key)
                if item.name == f"real-{prefix}"
            )
            proactive = ProactiveRepository(settings.proactive_database)
            proactive.commit_fetch(
                session_key=schedule_message.session_key,
                source_id="real-regression",
                result=ProactiveFetchResult(
                    (
                        ProactiveEvent(
                            "real-regression",
                            f"alert-{prefix}",
                            "alert",
                            datetime.now(UTC).isoformat(),
                            {
                                "title": f"【真实回归】主动链路 {prefix}",
                                "message": "请简短提醒用户主动链路已通过",
                            },
                        ),
                    )
                ),
                fetched_at=datetime.now(UTC),
            )
            if schedule.next_run_at is None:
                raise AssertionError("定时任务缺少下一次触发时间")
            delay = max(
                0.0,
                (schedule.next_run_at - datetime.now(UTC)).total_seconds(),
            )
            await asyncio.sleep(delay + 0.5)
            if await app.scheduler.run_once(now=datetime.now(UTC)) < 2:
                raise AssertionError("Scheduler 未发布定时与主动任务")
            if not await app.agent_loop.run_once() or not await app.agent_loop.run_once():
                raise AssertionError("AgentLoop 未执行定时与主动任务")
            schedule_state = _scalar(
                settings.operational_database,
                "SELECT state FROM scheduled_executions ORDER BY created_at DESC LIMIT 1",
            )
            proactive_state = _scalar(
                settings.proactive_database,
                "SELECT state FROM proactive_decisions ORDER BY decided_at DESC LIMIT 1",
            )
            if schedule_state != "succeeded" or proactive_state != "committed":
                raise AssertionError("定时或主动发送没有完成确认")
            print("PASS 定时任务 + 主动决策 + 飞书发送确认")

            _, status_reply = await _turn(
                app,
                queue,
                chat_id=target.chat_id,
                prefix=prefix,
                content="/memorystatus",
            )
            if "记忆" not in status_reply.content:
                raise AssertionError("内置状态插件未短路回复")
            print("PASS 内置插件状态命令 + 飞书")
        finally:
            await app.redis.flushdb()
            await app.close()


if __name__ == "__main__":
    asyncio.run(main())
