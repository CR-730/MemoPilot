from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from memopilot.persistence.migrations import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
from memopilot.runtime.contracts import FunctionCall
from memopilot.runtime.tools import ToolRegistry
from memopilot.scheduling.repository import ScheduleRepository
from memopilot.scheduling.service import ScheduleService
from memopilot.scheduling.tool_context import (
    bind_schedule_tool_context,
    reset_schedule_tool_context,
)
from memopilot.scheduling.tools import build_schedule_tools

RECEIVED_AT = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _registry(tmp_path):
    database = tmp_path / "operational.db"
    migrate_database(database, DatabaseKind.OPERATIONAL)
    with connect_database(database) as connection:
        now = RECEIVED_AT.isoformat()
        connection.execute(
            "INSERT INTO sessions(session_key, channel, chat_id, created_at, updated_at) "
            "VALUES ('feishu:chat-1', 'feishu', 'chat-1', ?, ?)",
            (now, now),
        )
        connection.execute(
            "INSERT INTO session_activity(session_key, activity_version, last_user_at, updated_at) "
            "VALUES ('feishu:chat-1', 1, ?, ?)",
            (now, now),
        )
    repository = ScheduleRepository(database)
    return ToolRegistry(build_schedule_tools(ScheduleService(repository))), repository


async def test_schedule_tool_uses_trusted_session_and_received_at(tmp_path) -> None:
    registry, repository = _registry(tmp_path)
    token = bind_schedule_tool_context("feishu:chat-1", received_at=RECEIVED_AT)
    try:
        observation = await registry.execute(
            FunctionCall(
                id="call-1",
                name="schedule",
                arguments={
                    "schedule_kind": "after",
                    "when": "30s",
                    "execution_mode": "instant",
                    "message": "喝水",
                    "name": "喝水提醒",
                },
            )
        )
    finally:
        reset_schedule_tool_context(token)

    assert observation.ok is True
    tasks = repository.list_for_session("feishu:chat-1")
    assert len(tasks) == 1
    assert tasks[0].next_run_at == RECEIVED_AT + timedelta(seconds=30)
    schedule_tool = registry.get_tool("schedule")
    assert schedule_tool is not None
    schedule_schema = schedule_tool.parameters
    assert "session_key" not in schedule_schema["properties"]
    assert "received_at" not in schedule_schema["properties"]


async def test_schedule_tool_localizes_naive_cli_timestamp(tmp_path) -> None:
    registry, repository = _registry(tmp_path)
    received_at = datetime(2026, 7, 21, 12, 0)
    token = bind_schedule_tool_context("feishu:chat-1", received_at=received_at)
    try:
        observation = await registry.execute(
            FunctionCall(
                id="naive-cli",
                name="schedule",
                arguments={
                    "schedule_kind": "after",
                    "when": "1m",
                    "execution_mode": "instant",
                    "message": "CLI 提醒",
                    "timezone": "Asia/Shanghai",
                },
            )
        )
    finally:
        reset_schedule_tool_context(token)

    assert observation.ok is True
    expected = received_at.replace(tzinfo=ZoneInfo("Asia/Shanghai")) + timedelta(minutes=1)
    assert repository.list_for_session("feishu:chat-1")[0].next_run_at == expected.astimezone(UTC)


async def test_schedule_tool_validates_mode_payload(tmp_path) -> None:
    registry, _ = _registry(tmp_path)
    token = bind_schedule_tool_context("feishu:chat-1", received_at=RECEIVED_AT)
    try:
        observation = await registry.execute(
            FunctionCall(
                id="call-2",
                name="schedule",
                arguments={
                    "schedule_kind": "after",
                    "when": "5m",
                    "execution_mode": "instant",
                },
            )
        )
    finally:
        reset_schedule_tool_context(token)

    assert observation.ok is False
    assert "message" in (observation.error_message or "")


async def test_list_and_cancel_tools_only_touch_current_session(tmp_path) -> None:
    registry, repository = _registry(tmp_path)
    token = bind_schedule_tool_context("feishu:chat-1", received_at=RECEIVED_AT)
    try:
        created = await registry.execute(
            FunctionCall(
                id="create",
                name="schedule",
                arguments={
                    "schedule_kind": "at",
                    "when": "2026-07-22T09:00:00+08:00",
                    "execution_mode": "agent",
                    "prompt": "查询天气",
                    "name": "天气",
                },
            )
        )
        listed = await registry.execute(
            FunctionCall(id="list", name="list_schedules", arguments={})
        )
        cancelled = await registry.execute(
            FunctionCall(
                id="cancel", name="cancel_schedule", arguments={"name": "天气"}
            )
        )
    finally:
        reset_schedule_tool_context(token)

    assert created.ok is True
    assert listed.ok is True
    assert listed.result["schedules"][0]["name"] == "天气"
    assert cancelled.result["cancelled_count"] == 1
    assert repository.list_for_session("feishu:chat-1") == ()


async def test_schedule_tools_require_runtime_context(tmp_path) -> None:
    registry, _ = _registry(tmp_path)

    observation = await registry.execute(
        FunctionCall(id="list", name="list_schedules", arguments={})
    )

    assert observation.ok is False
    assert "运行时上下文" in (observation.error_message or "")


async def test_schedule_tool_does_not_write_after_runtime_loses_fence(tmp_path) -> None:
    registry, repository = _registry(tmp_path)

    def lost_fence() -> None:
        raise RuntimeError("lost lease")

    token = bind_schedule_tool_context(
        "feishu:chat-1",
        received_at=RECEIVED_AT,
        assert_current=lost_fence,
    )
    try:
        observation = await registry.execute(
            FunctionCall(
                id="lost-fence",
                name="schedule",
                arguments={
                    "schedule_kind": "after",
                    "when": "5m",
                    "execution_mode": "instant",
                    "message": "不应创建",
                },
            )
        )
    finally:
        reset_schedule_tool_context(token)

    assert observation.ok is False
    assert repository.list_for_session("feishu:chat-1") == ()
