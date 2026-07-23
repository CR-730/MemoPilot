"""供 Agent 使用的定时任务工具。"""

from __future__ import annotations

from memopilot.runtime.tools import Tool
from memopilot.scheduling.contracts import ScheduledTask
from memopilot.scheduling.service import ScheduleService
from memopilot.scheduling.tool_context import (
    ScheduleToolContext,
    current_schedule_tool_context,
)


def build_schedule_tools(service: ScheduleService) -> tuple[Tool, Tool, Tool]:
    async def schedule_handler(
        schedule_kind: str,
        when: str,
        execution_mode: str,
        message: str | None = None,
        prompt: str | None = None,
        name: str | None = None,
        timezone: str | None = None,
    ) -> dict[str, object]:
        context = _require_context()
        if context.assert_current is not None:
            context.assert_current()
        task = service.schedule(
            session_key=context.session_key,
            received_at=context.received_at,
            schedule_kind=schedule_kind,
            when=when,
            execution_mode=execution_mode,
            message=message,
            prompt=prompt,
            name=name,
            timezone=timezone or context.default_timezone,
        )
        return _task_result(task)

    async def list_handler() -> dict[str, object]:
        context = _require_context()
        return {
            "schedules": [_task_result(task) for task in service.list(context.session_key)]
        }

    async def cancel_handler(
        task_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, object]:
        context = _require_context()
        if context.assert_current is not None:
            context.assert_current()
        cancelled = service.cancel(context.session_key, task_id=task_id, name=name)
        return {"cancelled_count": len(cancelled), "task_ids": list(cancelled)}

    return (
        Tool(
            name="schedule",
            description=(
                "创建用户定时任务。after 从当前用户消息到达时间起算；"
                "instant 到时发送固定文本，agent 到时重新执行提示。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "schedule_kind": {"type": "string", "enum": ["at", "after", "every"]},
                    "when": {
                        "type": "string",
                        "description": "at 用 HH:MM/ISO；after 用 5m；every 用 1h 或 cron",
                    },
                    "execution_mode": {"type": "string", "enum": ["instant", "agent"]},
                    "message": {"type": "string"},
                    "prompt": {"type": "string"},
                    "name": {"type": "string"},
                    "timezone": {"type": "string", "default": "Asia/Shanghai"},
                },
                "required": ["schedule_kind", "when", "execution_mode"],
                "additionalProperties": False,
            },
            handler=schedule_handler,
        ),
        Tool(
            name="list_schedules",
            description="列出当前会话仍启用的定时任务。",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=list_handler,
        ),
        Tool(
            name="cancel_schedule",
            description="按 task_id 或名称取消当前会话的定时任务。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
            handler=cancel_handler,
        ),
    )


def _require_context() -> ScheduleToolContext:
    context = current_schedule_tool_context()
    if context is None:
        raise RuntimeError("定时工具缺少可信运行时上下文")
    return context


def _task_result(task: ScheduledTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "name": task.name,
        "schedule_kind": task.schedule_kind,
        "when": task.schedule_expression,
        "execution_mode": task.execution_mode,
        "next_run_at": task.next_run_at.isoformat() if task.next_run_at else None,
        "timezone": task.timezone,
    }


__all__ = ["build_schedule_tools"]
