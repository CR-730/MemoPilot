"""将 Runtime 增量渲染为飞书实时过程卡。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Protocol

import httpx

from memopilot.channels.contracts import SendReceipt
from memopilot.channels.feishu import FeishuApiError
from memopilot.channels.feishu_cards import (
    ToolLiveLine,
    build_live_card,
    build_summary_card,
    format_tool_intent,
    format_tool_target,
)
from memopilot.runtime.contracts import FunctionCall, StreamDelta
from memopilot.runtime.tools import ToolObservation

logger = logging.getLogger(__name__)

_LIVE_STREAM_MIN_CHARS = 200
_LIVE_STREAM_MIN_INTERVAL_SECONDS = 2.0
_LIVE_MAX_FAILURES = 3
_LIVE_MAX_BACKOFF_SECONDS = 16.0
_LIVE_MAX_RATE_LIMITS = 5
_RATE_LIMIT_CODES = frozenset({99991400, 99991661, 230020, 230027, 11232})


class LiveCardTransport(Protocol):
    async def send_card(
        self,
        chat_id: str,
        content: str,
        *,
        provider_uuid: str,
    ) -> SendReceipt: ...

    async def patch_card(self, message_id: str, content: str) -> None: ...


class FeishuLiveProgress:
    """单个 Turn 的 live 状态；失败只降级预览，不改变最终回复结果。"""

    def __init__(
        self,
        transport: LiveCardTransport,
        *,
        chat_id: str,
        provider_uuid: str,
        authorize: Callable[[bool], bool] | None = None,
        min_interval_seconds: float = _LIVE_STREAM_MIN_INTERVAL_SECONDS,
        max_failures: int = _LIVE_MAX_FAILURES,
        max_rate_limits: int = _LIVE_MAX_RATE_LIMITS,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("live 最小刷新间隔不能小于 0")
        if max_failures < 1:
            raise ValueError("live 最大失败次数必须至少为 1")
        if max_rate_limits < 1:
            raise ValueError("live 最大限流次数必须至少为 1")
        self._transport = transport
        self._chat_id = chat_id
        self._provider_uuid = provider_uuid
        self._authorize = authorize or (lambda creating: True)
        self._min_interval = min_interval_seconds
        self._max_failures = max_failures
        self._max_rate_limits = max_rate_limits
        self._monotonic = monotonic or (lambda: asyncio.get_running_loop().time())
        self._thinking = ""
        self._reply = ""
        self._tools: list[ToolLiveLine] = []
        self._message_id: str | None = None
        self._next_at = 0.0
        self._last_length = 0
        self._failures = 0
        self._rate_limit_hits = 0
        self._interval = min_interval_seconds
        self._backoff_until = 0.0
        self._disabled = False
        self._lock = asyncio.Lock()

    async def on_stream_delta(self, delta: StreamDelta) -> None:
        if delta.content_delta:
            self._reply += delta.content_delta
        if delta.thinking_delta:
            self._thinking += delta.thinking_delta
        if self._message_id is None and not self._thinking.strip() and not self._tools:
            return
        live_length = len(self._reply) + len(self._thinking)
        now = self._monotonic()
        if now < self._next_at and live_length - self._last_length < _LIVE_STREAM_MIN_CHARS:
            return
        self._next_at = now + self._interval
        self._last_length = live_length
        await self._sync()

    async def on_tool_call_started(self, iteration: int, call: FunctionCall) -> None:
        del iteration
        self._tools.append(
            ToolLiveLine(
                call_id=call.id,
                tool_name=call.name,
                intent=format_tool_intent(call.arguments),
                target=format_tool_target(call.arguments),
            )
        )
        await self._sync()

    async def on_tool_call_completed(
        self,
        iteration: int,
        call: FunctionCall,
        observation: ToolObservation,
    ) -> None:
        del iteration
        line = next((item for item in self._tools if item.call_id == call.id), None)
        if line is None:
            line = ToolLiveLine(
                call_id=call.id,
                tool_name=call.name,
                intent=format_tool_intent(call.arguments),
                target=format_tool_target(call.arguments),
            )
            self._tools.append(line)
        line.status = "done" if observation.ok else "error"
        await self._sync()

    async def finalize(self) -> None:
        if self._disabled:
            return
        if self._message_id is None and not self._thinking.strip() and not self._tools:
            return
        summary = build_summary_card(self._thinking, self._tools)
        try:
            if self._message_id is None:
                if not self._is_authorized(creating=True):
                    return
                receipt = await self._transport.send_card(
                    self._chat_id,
                    summary,
                    provider_uuid=self._provider_uuid,
                )
                self._message_id = receipt.message_id
            else:
                if not self._is_authorized(creating=False):
                    return
                await self._transport.patch_card(self._message_id, summary)
        except Exception as exc:
            logger.warning("飞书过程卡定格失败，最终回复仍将独立发送: %s", exc)

    async def _sync(self) -> None:
        if self._disabled:
            return
        now = self._monotonic()
        if now < self._backoff_until:
            return
        card = build_live_card(self._thinking, self._tools, self._reply)
        async with self._lock:
            if self._disabled:
                return
            try:
                if self._message_id is None:
                    if not self._is_authorized(creating=True):
                        return
                    receipt = await self._transport.send_card(
                        self._chat_id,
                        card,
                        provider_uuid=self._provider_uuid,
                    )
                    self._message_id = receipt.message_id
                else:
                    if not self._is_authorized(creating=False):
                        return
                    await self._transport.patch_card(self._message_id, card)
            except Exception as exc:
                self._record_failure(exc)
                return
            self._failures = 0
            self._interval = self._min_interval

    def _record_failure(self, error: Exception) -> None:
        if _is_rate_limited(error):
            self._rate_limit_hits += 1
            if self._rate_limit_hits >= self._max_rate_limits:
                self._disabled = True
                logger.warning("飞书 live 连续限流达到预算，已降级关闭")
                return
            base = self._interval or _LIVE_STREAM_MIN_INTERVAL_SECONDS
            self._interval = min(
                max(base * 2, _retry_after_seconds(error)),
                _LIVE_MAX_BACKOFF_SECONDS,
            )
            self._backoff_until = self._monotonic() + self._interval
            logger.warning("飞书 live 卡片触发限流，退避 %.1f 秒", self._interval)
            return
        self._failures += 1
        if self._failures >= self._max_failures:
            self._disabled = True
        logger.warning(
            "飞书 live 卡片刷新失败 failures=%d disabled=%s: %s",
            self._failures,
            self._disabled,
            error,
        )

    def _is_authorized(self, *, creating: bool) -> bool:
        try:
            allowed = self._authorize(creating)
        except Exception as exc:
            logger.warning("飞书 live 权威状态校验失败，已降级关闭: %s", exc)
            allowed = False
        if not allowed:
            self._disabled = True
        return allowed


def _is_rate_limited(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429
    return (
        isinstance(error, FeishuApiError)
        and error.business_code in _RATE_LIMIT_CODES
    )


def _retry_after_seconds(error: Exception) -> float:
    if not isinstance(error, httpx.HTTPStatusError):
        return 0.0
    value = error.response.headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


__all__ = ["FeishuLiveProgress", "LiveCardTransport"]
