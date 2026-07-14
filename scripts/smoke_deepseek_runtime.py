"""使用真实 DeepSeek 手动验证一次含工具调用的 Turn。"""

from __future__ import annotations

import asyncio

from memopilot.config import MemoPilotSettings
from memopilot.runtime.engine import AgentRuntime, TurnInput
from memopilot.runtime.providers import OpenAICompatibleProvider
from memopilot.runtime.tools import Tool, ToolRegistry


async def _runtime_status(*, component: str) -> dict[str, str]:
    return {"component": component, "status": "ready"}


async def main() -> None:
    settings = MemoPilotSettings()
    api_key = settings.chat_api_key.get_secret_value()
    if not api_key:
        raise SystemExit("缺少 MEMOPILOT_CHAT_API_KEY，跳过真实 DeepSeek 冒烟测试")
    provider = OpenAICompatibleProvider.from_deepseek_credentials(
        api_key=api_key,
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_retry_limit,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    tools = ToolRegistry(
        [
            Tool(
                name="get_runtime_status",
                description="查询 MemoPilot 指定组件的当前状态",
                parameters={
                    "type": "object",
                    "properties": {
                        "component": {
                            "type": "string",
                            "description": "需要查询的组件名",
                        }
                    },
                    "required": ["component"],
                    "additionalProperties": False,
                },
                handler=_runtime_status,
            )
        ]
    )
    result = await AgentRuntime(
        provider,
        tools,
        max_iterations=settings.llm_max_iterations,
    ).run(
        TurnInput(
            session_key="smoke:deepseek",
            content=(
                "请先调用 get_runtime_status 查询 agent_runtime，"
                "再根据工具结果用一句中文告诉我状态。"
            ),
        )
    )
    if not result.react.tool_chain:
        raise RuntimeError("DeepSeek 未产生预期的工具调用")
    print(result.reply)
    print(
        f"phase_modules={len(result.phase_trace)} "
        f"tool_calls={len(result.react.tool_chain)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
