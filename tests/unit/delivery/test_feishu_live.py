from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import AsyncMock

import httpx
import pytest

from memopilot.channels.contracts import SendReceipt
from memopilot.channels.feishu import FeishuApiError
from memopilot.delivery.feishu_live import FeishuLiveProgress
from memopilot.runtime.contracts import FunctionCall, StreamDelta
from memopilot.runtime.tools import ToolObservation


@pytest.mark.asyncio
async def test_content_only_stream_does_not_create_empty_process_card() -> None:
    transport = AsyncMock()
    progress = FeishuLiveProgress(
        transport,
        chat_id="oc-chat",
        provider_uuid="live-uuid",
        min_interval_seconds=0,
    )

    await progress.on_stream_delta(StreamDelta(content_delta="直接回复"))
    await progress.finalize()

    transport.send_card.assert_not_awaited()
    transport.patch_card.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_progress_creates_updates_and_freezes_one_process_card() -> None:
    transport = AsyncMock()
    transport.send_card.return_value = SendReceipt(message_id="om-live")
    progress = FeishuLiveProgress(
        transport,
        chat_id="oc-chat",
        provider_uuid="live-uuid",
        min_interval_seconds=0,
    )
    call = FunctionCall(
        id="call-1",
        name="web_search",
        arguments={"description": "查资料", "query": "MemoPilot"},
    )

    await progress.on_stream_delta(StreamDelta(thinking_delta="先分析问题"))
    await progress.on_tool_call_started(1, call)
    await progress.on_tool_call_completed(
        1,
        call,
        ToolObservation(
            call_id="call-1",
            tool_name="web_search",
            ok=True,
            result="找到资料",
        ),
    )
    await progress.on_stream_delta(StreamDelta(content_delta="临时答案"))
    await progress.finalize()

    transport.send_card.assert_awaited_once()
    assert transport.send_card.await_args.kwargs["provider_uuid"] == "live-uuid"
    assert transport.patch_card.await_count >= 1
    final_card = json.loads(transport.patch_card.await_args.args[1])
    thinking = final_card["body"]["elements"][0]
    tools = final_card["body"]["elements"][1]["content"]
    assert thinking["tag"] == "collapsible_panel"
    assert thinking["expanded"] is False
    assert "先分析问题" in thinking["elements"][0]["content"]
    assert "web_search" in tools
    assert "✅" in tools


@pytest.mark.asyncio
async def test_live_progress_disables_preview_after_repeated_failures() -> None:
    transport = AsyncMock()
    transport.send_card.side_effect = RuntimeError("card unavailable")
    progress = FeishuLiveProgress(
        transport,
        chat_id="oc-chat",
        provider_uuid="live-uuid",
        min_interval_seconds=0,
        max_failures=3,
    )

    for delta in ("一", "二", "三", "四"):
        await progress.on_stream_delta(StreamDelta(thinking_delta=delta))
    await progress.finalize()

    assert transport.send_card.await_count == 3
    transport.patch_card.assert_not_awaited()


def _http_429() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://open.feishu.cn/open-apis/im/v1/messages")
    response = httpx.Response(429, request=request, headers={"Retry-After": "3"})
    return httpx.HTTPStatusError("rate limited", request=request, response=response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_factory",
    [
        _http_429,
        lambda: FeishuApiError(99991400, "rate limited"),
    ],
)
async def test_rate_limit_uses_bounded_backoff_then_disables_live(
    error_factory: Callable[[], Exception],
) -> None:
    now = [0.0]
    transport = AsyncMock()
    transport.send_card.side_effect = error_factory()
    progress = FeishuLiveProgress(
        transport,
        chat_id="oc-chat",
        provider_uuid="live-uuid",
        min_interval_seconds=0,
        max_rate_limits=2,
        monotonic=lambda: now[0],
    )

    await progress.on_stream_delta(StreamDelta(thinking_delta="第一次"))
    now[0] = 20
    await progress.on_stream_delta(StreamDelta(thinking_delta="第二次"))
    now[0] = 40
    await progress.on_stream_delta(StreamDelta(thinking_delta="不再重试"))

    assert transport.send_card.await_count == 2
