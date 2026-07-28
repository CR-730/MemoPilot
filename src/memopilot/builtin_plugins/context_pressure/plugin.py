from __future__ import annotations

from memopilot.extensions.plugin_base import Plugin
from memopilot.extensions.plugin_events import AfterStepCtx


class ContextPressureStopModule:
    slot = "context_pressure.stop"
    requires = ("after_step.copy_input", "step:ctx")
    produces = ("step:ctx",)

    async def run(self, frame: object) -> object:
        slots = getattr(frame, "slots", None)
        if not isinstance(slots, dict):
            return frame
        ctx = slots.get("step:ctx")
        if not isinstance(ctx, AfterStepCtx) or not ctx.has_more:
            return frame
        if (
            ctx.context_tokens_estimate
            <= ctx.context_window_tokens * 80 // 100
        ):
            return frame
        ctx.early_stop = True
        ctx.early_stop_reason = "context_pressure"
        return frame


class ContextPressurePlugin(Plugin):
    name = "context_pressure"
    version = "0.1.0"
    desc = "上下文压力过高时请求被动循环阶段性收尾"

    def after_step_modules(self) -> list[object]:
        return [ContextPressureStopModule()]
