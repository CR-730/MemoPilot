"""从可后台运行的 Skills 中选择一次 Drift 任务。"""

# 原型 Drift 提示词按运行合同保留。
# ruff: noqa: E501

from __future__ import annotations

from memopilot.extensions.skills import SkillCatalog
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.providers import ChatProvider

DRIFT_SYSTEM_PROMPT = """你现在有一段空闲时间（Drift 模式）。没有外部内容需要推送，
你可以自主决定做一件有意义的事。本轮记忆、skill 和工作区信息会在后续上下文中提供。

【执行规则】
1. 每次进入 Drift 都先重新比较所有可用 skill；不要因为某个 skill 最近刚运行过，或 next 很明确，就默认继续它。
2. 自主选择一个 skill，read_file 读取它的 SKILL.md；再读取该 skill 的 working files 了解进度。
3. 读完后执行当前最直接的下一步动作，不要只因为看到 queue、next 或等待描述就立刻 finish_drift。
4. 如果 skill 明显处于等待用户回复或外部条件，改选其他 skill。
5. 只有完成明确动作，或确认当前确实无事可做，才允许 finish_drift。
6. 有价值的发现立即 write_file 或 edit_file。
7. message_push 的表达必须像自然想到的一句聊天，而不是汇报内部流程。
8. 单次 run 最多只能 message_push 一次。
9. message_push 成功后禁止 recall_memory、web_fetch、web_search、fetch_messages、search_messages、shell；
   后续只允许 write_file、edit_file 和 finish_drift。
10. 结束前必须调用 finish_drift 保存状态，并用 message_result 标注 sent 或 silent；
    成功 message_push 时必须是 sent，否则必须是 silent。

【可用工具】
read_file, write_file, edit_file, recall_memory, web_fetch, web_search,
fetch_messages, search_messages, shell, message_push, finish_drift。
"""

_SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_drift_skill",
        "description": "选择本次空闲时间最值得执行的一项后台 Skill。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "必须来自候选列表的 Skill 名称。",
                }
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        },
    },
}


class DriftSkillSelector:
    def __init__(self, provider: ChatProvider, skills: SkillCatalog) -> None:
        self._provider = provider
        self._skills = skills

    async def select(self) -> str:
        candidates = self._skills.background_candidates()
        if not candidates:
            raise ValueError("没有可用的后台 Skill")
        names = {skill.name for skill in candidates}
        catalog = "\n".join(
            f"- {skill.name}: {skill.description}" for skill in candidates
        )
        response = await self._provider.complete(
            messages=(
                ChatMessage.system(
                    "你负责选择一个空闲时执行的后台任务。"
                    "只允许调用 select_drift_skill，不要输出自然语言。"
                ),
                ChatMessage.user(f"可选 Skills：\n{catalog}"),
            ),
            tools=(_SELECT_TOOL,),
        )
        calls = tuple(
            call for call in response.tool_calls if call.name == "select_drift_skill"
        )
        if len(calls) != 1 or len(response.tool_calls) != 1:
            raise ValueError("Drift 选择必须恰好调用一次 select_drift_skill")
        selected = str(calls[0].arguments.get("skill_name") or "").strip()
        if selected not in names:
            raise ValueError(f"Drift Skill 不可用: {selected}")
        return selected


__all__ = ["DRIFT_SYSTEM_PROMPT", "DriftSkillSelector"]
