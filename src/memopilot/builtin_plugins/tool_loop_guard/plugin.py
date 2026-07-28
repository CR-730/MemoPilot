from __future__ import annotations

import json
from dataclasses import dataclass

from memopilot.extensions.decorators import on_tool_pre
from memopilot.extensions.hooks import HookOutcome
from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import PreToolCtx

_REPEAT_LIMIT = 3
_DENY_PREFIX = "tool_loop_guard:"
_EXCLUDED_TOOLS = frozenset({"task_output", "task_stop"})


@dataclass
class _LoopState:
    signature: str = ""
    repeat_count: int = 0


class ToolLoopGuard(Plugin):
    name = "tool_loop_guard"
    version = "0.1.0"
    desc = "检测连续重复的工具调用并提前截断"

    def __init__(self) -> None:
        self._states: dict[str, _LoopState] = {}

    @on_tool_pre()
    async def detect_repeated_tool_call(
        self,
        event: PreToolCtx,
    ) -> HookOutcome | None:
        signature, active_index = self._event_signature(event)
        if not signature or event.tool_name in _EXCLUDED_TOOLS:
            return None
        state_key = self._state_key(event)
        state = self._states.get(state_key)
        if (event.tool_batch_index or 0) == active_index:
            state = self._states.setdefault(state_key, _LoopState())
            if signature == state.signature:
                state.repeat_count += 1
            else:
                state.signature = signature
                state.repeat_count = 1
        elif state is None or signature != state.signature:
            return None
        if state.repeat_count < _REPEAT_LIMIT:
            return None
        return HookOutcome(
            decision="deny",
            reason=(
                f"{_DENY_PREFIX}连续重复调用工具 "
                f"{state.repeat_count} 次，已截断并进入收尾。"
            ),
        )

    def _state_key(self, event: PreToolCtx) -> str:
        if event.session_key:
            return f"{event.source}:{event.session_key}"
        return f"{event.source}:{event.channel}:{event.chat_id}"

    def _signature(self, tool_name: str, arguments: dict[str, object]) -> str:
        args = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return f"{tool_name}:{args}"

    def _event_signature(self, event: PreToolCtx) -> tuple[str, int]:
        if not event.tool_batch:
            if event.tool_name in _EXCLUDED_TOOLS:
                return "", 0
            return self._signature(event.tool_name, event.arguments), 0

        parts: list[str] = []
        active_index = -1
        for index, raw_call in enumerate(event.tool_batch):
            if not isinstance(raw_call, dict):
                continue
            tool_name = str(raw_call.get("tool_name") or "")
            if tool_name in _EXCLUDED_TOOLS:
                continue
            arguments = raw_call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            if active_index < 0:
                active_index = index
            parts.append(self._signature(tool_name, arguments))
        if active_index < 0:
            return "", 0
        return "|".join(parts), active_index
