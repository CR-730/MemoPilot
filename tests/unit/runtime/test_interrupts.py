from __future__ import annotations

from memopilot.runtime.contracts import FunctionCall, StreamDelta
from memopilot.runtime.interrupts import InterruptProgressRecorder, render_resumed_message
from memopilot.runtime.tools import ToolObservation
from memopilot.tasks.operational import TurnInterruptSnapshotRecord


async def test_progress_recorder_preserves_partial_reply_thinking_and_tools() -> None:
    recorder = InterruptProgressRecorder()
    call = FunctionCall(id="c1", name="weather", arguments={"city": "上海"})

    await recorder.on_stream_delta(StreamDelta(thinking_delta="先查天气"))
    await recorder.on_tool_call_started(1, call)
    await recorder.on_tool_call_completed(
        1,
        call,
        ToolObservation(
            call_id="c1",
            tool_name="weather",
            ok=True,
            result={"temperature": 30},
        ),
    )
    await recorder.on_stream_delta(StreamDelta(content_delta="天气已查到"))

    snapshot = recorder.snapshot(original_message="查天气后发邮件")
    assert snapshot.partial_thinking == "先查天气"
    assert snapshot.partial_reply == "天气已查到"
    assert snapshot.tools_used == ("weather",)
    assert snapshot.tool_chain[-1]["status"] == "done"


def test_resumed_message_matches_prototype_semantics() -> None:
    snapshot = TurnInterruptSnapshotRecord(
        snapshot_id="snapshot-1",
        source_run_id="run-1",
        session_key="feishu:chat-1",
        original_message="查天气后发邮件",
        partial_reply="天气已查到",
        partial_thinking="下一步准备发邮件",
        tools_used=("weather",),
        tool_chain=({"tool": "weather", "status": "done"},),
    )

    message = render_resumed_message(snapshot, "改成发给小王")

    assert "上一轮任务" in message
    assert "查天气后发邮件" in message
    assert "上一轮中间结果" in message
    assert "天气已查到" in message
    assert "weather" in message
    assert "用户补充要求" in message
    assert "改成发给小王" in message
