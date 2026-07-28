from __future__ import annotations

from typing import Any

from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import BeforeTurnCtx, BeforeTurnInput


class SetupHelperModule:
    slot = "setup_helper.chatid"
    requires = ("before_turn.build_ctx", "session:ctx")
    produces = ("session:ctx",)

    async def run(self, frame: Any) -> Any:
        state = frame.input
        ctx = frame.slots.get("session:ctx")
        if not isinstance(state, BeforeTurnInput) or not isinstance(ctx, BeforeTurnCtx):
            return frame
        command = state.content.strip().split(maxsplit=1)
        if not command:
            return frame
        name = command[0].lower().split("@", 1)[0]
        if name not in {"/chatid", "/myid"}:
            return frame
        ctx.abort = True
        ctx.abort_reply = _reply(state.channel, state.chat_id)
        return frame


class SetupHelperPlugin(Plugin):
    name = "setup_helper"

    def before_turn_modules(self) -> list[object]:
        return [SetupHelperModule()]


def _reply(channel: str, chat_id: str) -> str:
    return "\n".join(
        [
            f"当前 channel：`{channel or '（未知）'}`",
            f"当前 chat_id：`{chat_id or '（未知）'}`",
            "",
            "该私聊会话已自动登记，无需手工配置主动推送目标。",
            "如需开启主动能力，请在 config.toml 中设置：",
            "",
            "```toml",
            "[proactive]",
            "enabled = true",
            "```",
            "",
            "主动内容来源请在工作区的 proactive_sources.json 中配置。",
        ]
    )
