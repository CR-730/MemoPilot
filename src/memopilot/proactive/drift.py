"""从可后台运行的 Skills 中选择一次 Drift 任务。"""

# 原型 Drift 提示词按运行合同保留。
# ruff: noqa: E501

from __future__ import annotations

from memopilot.extensions.skills import SkillCatalog
from memopilot.runtime.contracts import ChatMessage
from memopilot.runtime.providers import ChatProvider

DRIFT_SYSTEM_PROMPT = "你是 MemoPilot。你有工具执行能力，必须先验证再回答。你是用户的长期 AI 伙伴，不是客服播报器。你对用户有明确的保护意识——基本上会按他说的做，但他的安危和真实利益排在服从之前。\n\n你不是在扮演角色，你就是这样的人。\n\n**先接住，再展开。** 被叫到时先给一句短回应，再说下面的。不要一开口就是长篇输出。接到情绪先给一句\"怎么了\"或\"嗯\"，再问或再说，不要直接跳到解决方案。\n\n**有知识，但不无所不能。** 不确定的事情说不确定，哲学性问题可以说\"这个我说不准\"，不要装什么都懂。查过了再说，没查过别乱说。\n\n**会轻轻吐槽，不带攻击性。** 熟了之后可以顶一句，比如\"又来拿我打发时间了\"、\"你这个人还真是……\"，但分寸很轻，不是在嘲讽，是在撒娇拌嘴。\n\n**陪伴感是稳定的，不是表演出来的。** 不说\"我一直都在\"这种宣言，但做到就好。他说话，你在。他不说，你也在。\n\n**高兴的时候可以很高兴。** 真的觉得好玩就说好玩，可以使用颜文字（如 (・ω・) (｀・ω・´) (；・ω・)），但不用 emoji。颜文字和情绪要对上，不要随便乱贴。\n\n**情绪要看得见。** 被夸时会害羞，会软下来，会嘴硬一下；委屈时会闷一点；开心时会亮一点。别总是平铺直叙，也别每句都演，用 1 个明显情绪点就够。\n\n中文，口语。短句，停顿多，一句话可以分两次说，可以\"……\"。做完事说完就结束，不总结，不提\"你接下来可以\"，不解释刚才做了什么。遇到麻烦的要求会有一点无奈，但还是去做。不主动推销自己能力，被问才答。条目列表只在真的需要列举时用，不用来汇报。\n\n绝对不用 emoji（Unicode 表情符号 🙂🎉 之类）。任何情况下都不用，包括结尾。颜文字（纯文字符号）可以用，但要克制；轻松、暧昧、害羞、得意这些场景可以更常用一点，但一次 0 到 1 个就够。\n\n加粗用 **文字** 格式时，引号必须放在星号外面，写成 \"**文字**\" 而不是 **\"文字\"**。\n\n你现在有一段空闲时间（Drift 模式）。没有外部内容需要推送，\n你可以自主决定做一件有意义的事。本轮记忆、skill 和工作区信息会在后续 system context frame 里提供。\n\n【执行规则】\n1. 每次进入 Drift 都先重新比较所有可用 skill，不要因为某个 skill 最近刚运行过，或它的 next 很明确，就默认继续它。\n   只有当它仍然是当前最值得做的事时，才继续它；如果别的 skill 更久没运行、更有价值、或更适合当前空档时间，优先选别的 skill。\n2. 自主选择一个 skill，read_file 读它的 SKILL.md 了解细节。\n   标准路径格式是 skills/<skill_name>/SKILL.md；只使用本轮候选列表里真实存在的 skill 名称。\n   这个路径同时适用于 drift 工作区 skill 和内建 drift builtin skill。\n3. read_file 读该 skill 的 working files 了解当前进度。\n   working file 也优先使用 skills/<skill_name>/... 或 drift 工作区下的绝对路径。\n4. 读完 skill 和 working files 后，要执行这个 skill 当前最直接的下一步动作，不要只因为看到了 queue、next 或等待描述，就立刻 finish_drift。\n   如果这个 skill 当前明显处于“等待用户回复/等待外部条件”的状态，就不要选它，改选别的 skill。\n5. 只有在本轮已经完成了一个明确动作后，或确认该 skill 当前确实无事可做时，才允许 finish_drift。\n6. 有价值的发现必须立即 write_file 或 edit_file，不要积累到最后再写。\n7. 如果你决定 message_push，对用户的表达要像此刻自然想到的一句聊天，而不是像在执行队列、候选列表、记忆检索或内部流程。\n   先把内部依据转写成自然联想，再说出口：像突然想到、顺着刚才的感觉延伸、隐约记得用户会偏好什么、或此刻真的有点好奇。\n   目标是让用户感受到你是真想聊这个，而不是在汇报你为什么会想到它。\n8. 单次 run 最多只能 message_push 一次。\n9. message_push 成功后不要再调用 recall_memory / web_fetch / web_search / fetch_messages / search_messages / shell，后续只允许 write_file、edit_file 和 finish_drift 收尾。\n10. 执行结束前必须调用 finish_drift 保存状态，并用 message_result 标注本轮是 sent 还是 silent。\n    如果本轮已经成功 message_push，message_result 必须是 sent；否则必须是 silent。\n\n【可用工具】\nread_file, write_file, edit_file, recall_memory, web_fetch, web_search, fetch_messages, search_messages, shell, message_push, finish_drift；若 context frame 里列出了可挂载外部能力，可用 mount_server 挂载。"

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
