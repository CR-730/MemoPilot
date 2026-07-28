from __future__ import annotations

from typing import Any

from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import BeforeTurnCtx, BeforeTurnInput


class UndoCommandModule:
    slot = "plugin_undo.undo"
    requires = ("before_turn.build_ctx", "session:ctx")
    produces = ("session:ctx",)

    def __init__(self, repository: object) -> None:
        self.repository = repository

    async def run(self, frame: Any) -> Any:
        state = frame.input
        ctx = frame.slots.get("session:ctx")
        if not isinstance(state, BeforeTurnInput) or not isinstance(ctx, BeforeTurnCtx):
            return frame
        command = state.content.strip().split(maxsplit=1)
        if not command or command[0].lower().split("@", 1)[0] != "/undo":
            return frame
        undo = getattr(self.repository, "undo_last_turn", None)
        if not callable(undo):
            reply = "撤销失败：会话仓储不可用。"
        else:
            try:
                result = undo(state.session_key)
            except Exception:
                reply = "撤销失败：会话数据未修改。"
            else:
                reply = (
                    "没有可撤销的上一轮对话。"
                    if result is None
                    else (
                        "已撤销上一轮对话。"
                        f"\n删除消息：{result[0]} 条"
                        f"\n整理游标：{result[1]} → {result[2]}"
                        "\n记忆回滚：未执行（当前数据无法可靠关联本轮已提炼记忆）。"
                    )
                )
        ctx.abort = True
        ctx.abort_reply = reply
        return frame


class PluginUndo(Plugin):
    name = "plugin_undo"

    def before_turn_modules(self) -> list[object]:
        return [UndoCommandModule(self.context.session_manager)]
