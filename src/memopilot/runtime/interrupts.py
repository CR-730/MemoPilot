"""原型式 Turn 中断进度快照与语义续接。"""

from __future__ import annotations

import json
from collections.abc import Iterable

from memopilot.runtime.contracts import FunctionCall, StreamDelta
from memopilot.runtime.memory_citations import visible_response_prefix
from memopilot.runtime.react import ReActProgressObserver
from memopilot.runtime.tools import ToolObservation
from memopilot.tasks.operational import (
    TurnInterruptSnapshot,
    TurnInterruptSnapshotRecord,
)


class InterruptProgressRecorder:
    def __init__(self) -> None:
        self._reply = ""
        self._thinking = ""
        self._tools: list[str] = []
        self._tool_chain: list[dict[str, object]] = []

    async def on_stream_delta(self, delta: StreamDelta) -> None:
        self._reply += delta.content_delta
        self._thinking += delta.thinking_delta

    async def on_tool_call_started(self, iteration: int, call: FunctionCall) -> None:
        if call.name not in self._tools:
            self._tools.append(call.name)
        self._tool_chain.append(
            {
                "iteration": iteration,
                "call_id": call.id,
                "tool": call.name,
                "arguments": call.arguments,
                "status": "running",
            }
        )

    async def on_tool_call_completed(
        self,
        iteration: int,
        call: FunctionCall,
        observation: ToolObservation,
    ) -> None:
        del iteration
        item = next(
            (entry for entry in reversed(self._tool_chain) if entry["call_id"] == call.id),
            None,
        )
        if item is None:
            await self.on_tool_call_started(0, call)
            item = self._tool_chain[-1]
        item["status"] = "done" if observation.ok else "error"
        item["observation"] = json.loads(observation.content)

    def snapshot(self, *, original_message: str) -> TurnInterruptSnapshot:
        return TurnInterruptSnapshot(
            original_message=original_message,
            partial_reply=visible_response_prefix(self._reply),
            partial_thinking=self._thinking,
            tools_used=tuple(self._tools),
            tool_chain=tuple(dict(item) for item in self._tool_chain),
        )


class CompositeProgressObserver:
    def __init__(self, observers: Iterable[ReActProgressObserver]) -> None:
        self._observers = tuple(observers)

    async def on_stream_delta(self, delta: StreamDelta) -> None:
        for observer in self._observers:
            try:
                await observer.on_stream_delta(delta)
            except Exception:
                continue

    async def on_tool_call_started(self, iteration: int, call: FunctionCall) -> None:
        for observer in self._observers:
            try:
                await observer.on_tool_call_started(iteration, call)
            except Exception:
                continue

    async def on_tool_call_completed(
        self,
        iteration: int,
        call: FunctionCall,
        observation: ToolObservation,
    ) -> None:
        for observer in self._observers:
            try:
                await observer.on_tool_call_completed(iteration, call, observation)
            except Exception:
                continue


def render_resumed_message(
    snapshot: TurnInterruptSnapshotRecord,
    user_message: str,
) -> str:
    middle = (
        visible_response_prefix(snapshot.partial_reply).strip()
        or snapshot.partial_thinking.strip()
        or "暂无"
    )
    tools = "、".join(snapshot.tools_used) or "无"
    return (
        "[中断任务续接]\n"
        f"上一轮任务：\n{snapshot.original_message}\n\n"
        f"上一轮中间结果：\n{middle}\n\n"
        f"已使用工具：{tools}\n\n"
        f"用户补充要求：\n{user_message}\n\n"
        "请基于已有结果继续完成任务；不要无理由重复已完成的工具操作。"
    )


__all__ = [
    "CompositeProgressObserver",
    "InterruptProgressRecorder",
    "render_resumed_message",
]
