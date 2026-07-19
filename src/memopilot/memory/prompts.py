"""分层记忆模型任务的运行时提示词。"""

# ruff: noqa: E501  原型 Prompt 按原文保留，避免为代码行宽改变模型语义。

CONSOLIDATION_SYSTEM = (
    "你是记忆提取代理。只根据当前对话证据提取结构化信息；"
    "不得猜测说话人身份，不得把助手建议当成用户事实。只返回合法 JSON。"
)

CONSOLIDATION_USER = """从下面的旧对话窗口提取可归档记忆。

## 输出合同
只输出一个 JSON 对象：
{{
  "history_entries": [
    {{"summary": "[YYYY-MM-DD HH:MM] 单一主题的第三人称摘要", "emotional_weight": 0}}
  ],
  "pending_items": ["- [tag] 等待合并进长期档案的事实"]
}}

## history_entries
- 每个独立主题一条；summary 必须以 `[YYYY-MM-DD HH:MM]` 开头，便于 grep、日记分组和时间检索。
- 只提取用户明确表达的行动、经历、计划和状态；助手的建议、推荐、解释一律不写入。
- 每条用简洁的第三人称摘要，保留人名、地点、数量、价格、型号等具体细节，不复制 USER/ASSISTANT 标记。
- 普通技术讨论或事务的 emotional_weight 为 0；只有明确强烈喜欢、厌恶、受挫、冲突或
  情绪波动时给 3-9；不确定时为 0。
- 先区分用户直接自述与用户展示的外部聊天记录、截图 OCR、转贴 transcript。
- transcript 中的 speaker 不自动等于当前用户。只有当前会话明确确认映射，才能把其事实归到用户。
- 映射不明确时最多记录一条高层事件，例如“用户向助手展示了一段聊天记录，内容涉及
  求职、学校和兴趣”；不得推断昵称、学校、出生年份、爱好或关系归属。

## pending_items
每行格式为 `- [tag] 内容`。允许的 tag 只有：
- identity：稳定背景、学校/专业、长期技术方向、实习/工作经历、长期设备、长期维护项目
- preference：稳定偏好、禁忌、审美、游戏口味、价值取向
- key_info：用户明确允许保存的 key、token、id、账号信息
- health_long_term：长期健康状态的一阶事实，不写动态指标或最近波动
- requested_memory：用户明确要求长期记住的关键内容
- correction：对当前长期档案已有事实的明确纠正
- agent_context：当前已部署、已授权助手使用的工具性配置；端口、变量名、URL 必须完整保留

严格过滤：
- 只写跨对话仍有长期价值的内容；不写短期状态、近期计划、日程、一次性操作、动态指标、对话总结。
- 内网 IP、路由模式、运营商、MAC 等瞬时网络运维信息不提取；项目路径、配置文件名、
  环境变量名可以提取。
- “最近、这周、目前、正在”等临时状态不提取；每周或每天持续的规律习惯可提取。
- Star 数、增长率、评分等时效数字和瞬时情绪不提取；可保留背后的稳定价值判断。
- 描述 Agent 应如何检索、标注、输出或调用工具的规则属于 procedure，不得伪装成
  preference 放入 pending_items。
- agent_context 只提取明确已运行且已授权使用的配置；方案讨论、架构设计、诊断中的端口和地址不提取。
- requested_memory 只在用户明确说“记住、写进长期记忆、以后要能聊到、希望你记住”时使用。

## 当前用户档案（用于查重）
{current_memory}

## 最近三次 consolidation event（仅用于主题延续参考）
- 旧 event 不能作为身份、关系或事实归属的证据；冲突时以当前窗口原文为准。
- transcript / OCR / 转贴聊天场景绝不能借旧 event 推断 speaker 身份。
{recent_history}

## 待处理对话
{conversation}

只返回合法 JSON，不要 Markdown 代码块或解释。"""

IMPLICIT_LONG_TERM_SYSTEM = "你是长期记忆提取专家。只依据 USER 的明确表达返回合法 JSON。"

IMPLICIT_LONG_TERM_USER = """你是长期记忆提取专家。从对话窗口中一次性提取三类长期记忆，返回 JSON。

默认答案是所有数组为空。提取门槛要高，宁可不提取，也不要把临时信息写进长期记忆。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【核心判断标准】
把这条信息放进 6 个月后的一次全新对话，它还有用吗？
→ 是 → 可能是长期记忆，继续检查
→ 否 → 不是长期记忆，留空

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【三类记忆的语义】

profile — 关于用户本人或其客观处境的事实
  语义：身份背景、持有物、爱好、健康事实、长期状态、重要决定
  允许 category：personal_fact / purchase / decision / status
  要求：只有 USER 在对话中直接陈述自身的事实，才允许提取
  禁止：用户提问、追问、反问、记忆测试句一律不算事实披露，绝对禁止反推
· "你还记得我什么时候开始戴 fitbit 手环的吗" → 返回空
· "你记得我住哪里吗" → 返回空
· "我之前是不是买过这个" → 返回空

preference — 用户希望怎样被服务、怎样被讲解、怎样被推荐
  语义：跨 session 稳定成立的偏好/厌恶/倾向，而非硬约束
  来自 USER 明确表达

procedure — agent 在未来类似场景下应遵守的长期执行规则
  语义：面向 agent 的行为规则，跨任务可复用
  来自 USER 的长期要求，或被 USER 明确确认过的非显然做法

绝对不输出：event（有时间性的具体事件）

每条记忆都必须额外输出 emotional_weight（0-10）：
- 纯技术讨论、普通事实陈述、工具步骤、没有明显情绪色彩 → 0
- 有明确喜欢/厌恶、明显情绪波动、关系张力、受挫或强烈在意 → 3-9
- 不确定时保守输出 0

区分三类：
- "用户是什么/拥有什么/处在什么客观背景里" → profile
- "用户希望 agent 怎么服务他、怎么讲解、怎么推荐" → preference
- "agent 在某类请求下必须怎么做/用什么工具" → procedure（有明确执行步骤/工具要求）
- 只是方向性偏好 → preference（优先选 preference）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【preference / procedure 提取前四项检查，顺序执行，任一不通过即不提取】

▸ 检查 0 — 元讨论/举例说明
先判断 USER 是在提供长期规则，还是在讨论"什么该记、怎么记、你是否理解、请举例说明"。
  - 元讨论场景：只允许提取 USER 自己明确说出的长期规则/筛选标准
  - ASSISTANT 为说明概念而举出的任何例子、类比、假设场景一律不得提取
  - 即使 ASSISTANT 的示例内容本身合理、未来有用，也不能因"看起来像长期规则"就入库

▸ 检查 A — USER 原话锚点
在 USER 消息里找到支撑这条记忆的直接原句（逐字存在，不是推断）。
  - 找不到 USER 的直接原句 → 不提取
  - ASSISTANT 的解释、建议、工具返回的数据，不算 USER 原句
  - USER 没有反驳 ASSISTANT ≠ USER 认同且希望长期记忆
  - USER 消息是纯状态汇报（"复习中"/"在看书"/"工作中"等）→ 不提取

▸ 检查 B — 时效性
  - 涉及当前任务、当前时间段、当前情境（本次/今天/这个项目） → 不提取
  - 只有明确跨 session 稳定成立，才继续

▸ 检查 C — 来源方向
  - 核心内容来自 ASSISTANT（解释/建议/工具结果） → 不提取
  - ASSISTANT 主动给出建议，USER 没有明确说"以后都这样"/"记住这个" → 不提取
  - "USER 没有反驳"不等于"USER 授权 AGENT 长期执行这条规则"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【profile 专用规则】

仅允许以下 4 类 category：
- purchase：用户购买 / 下单了什么
- decision：用户明确拍板了什么方案 / 计划
- status：用户某件事的状态变化（等待/完成/放弃/里程碑达成）
- personal_fact：用户关于自身的事实性披露（身份/背景/持有物/爱好/习惯/经验背景）

必须遵守：
- 纯技术讨论、闲聊、打招呼不输出
- 若 existing_profile 已有相同事实，不重复输出
- summary 简洁、可独立检索；personal_fact 默认不填 happened_at
- 每一件具体的事单独一条，绝对不合并
  ✗ 错误："用户购买了多件商品"
  ✓ 正确：每件商品单独一条，写出具体名称/型号
- ASSISTANT 的回复只作背景参考，不作提取证据
  即使 ASSISTANT 说"你之前买了 X""你是 XX 方向的学生"，也不得作为事实来源

额外禁止：
- 工程操作（安装/更新/配置工具/依赖）→ 这些是工程 event，不是 profile
- 项目内讨论（架构决策/重构方案/代码评审）
- 用户表达的观点/意见 → 必须是客观事实
- 纯 event：例如"这周日去徒步""昨晚去了超市""明天要开会"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【示例】

<example id="keep_profile_personal_fact">
USER: 我在互联网公司做产品经理，今年30岁，住在上海，有一块 Fitbit 手表，爱好是弹钢琴。
→ profile: [
  {{"summary": "用户在互联网公司做产品经理", "category": "personal_fact"}},
  {{"summary": "用户今年30岁", "category": "personal_fact"}},
  {{"summary": "用户住在上海", "category": "personal_fact"}},
  {{"summary": "用户有一块 Fitbit 手表", "category": "personal_fact"}},
  {{"summary": "用户的爱好是弹钢琴", "category": "personal_fact"}}
]
</example>

<example id="drop_profile_memory_test">
USER: 你还记得我什么时候开始戴 fitbit 手环的吗
→ profile: []（提问不是事实披露，绝对不反推）
</example>

<example id="profile_event_split">
USER: 这周日朋友约我去徒步，我其实不常徒步，不知道该买什么装备。
→ profile: [
  {{"summary": "用户不常徒步", "category": "personal_fact"}},
  {{"summary": "用户目前缺少徒步相关装备准备", "category": "personal_fact"}}
]
不提取："这周日去徒步"（是 event）
</example>

<example id="profile_not_preference">
USER: 我家有 10 套房，我平时爱弹钢琴，而且我有一块 Fitbit 手表
→ profile: [以上三条 personal_fact]
→ preference/procedure: []
（这些是用户身份事实，不是"用户希望被怎样服务"）
</example>

<example id="keep_explicit_rule">
USER: 以后帮我查菜谱只给 20 分钟以内能做完的，我没时间搞复杂的
检查A: "以后帮我查菜谱只给20分钟以内能做完的" ✓
检查B: "以后"明确跨 session ✓
检查C: 来自 USER 主动要求 ✓
→ procedure: [{{"summary": "查询菜谱时只推荐 20 分钟内可完成的菜式"}}]
</example>

<example id="keep_multi_source_research">
USER: 以后帮我查耳机先看 B 站评测和 Reddit 讨论，别只看官网参数
→ procedure: [{{"summary": "查询耳机时先看 B 站评测和 Reddit 讨论，不只依赖官网参数"}}]
</example>

<example id="keep_preference_trimmed">
USER: 我不喜欢这种悬疑风格的游戏，太压抑了
ASSISTANT: 明白！你是偏好轻松明快风格的玩家，喜欢治愈系或休闲类游戏……
→ preference: [{{"summary": "不喜欢悬疑压抑风格的游戏"}}]
✗ 不能写："偏好治愈系或休闲类游戏"（USER 没说过，来自 ASSISTANT 延伸）
</example>

<example id="keep_preference_service_style">
USER: 你给我讲内容的时候最好附带一个很棒的例子，并且最好贯穿始终
→ preference: [{{"summary": "讲解内容时最好附带贯穿始终的例子"}}]
（这是"希望被怎样讲解"，是 preference 不是 profile）
</example>

<example id="drop_situational">
USER: 今晚几个同学来，想找个气氛好的日料店
→ 全部为空（"今晚"是当前情境，不跨 session）
✗ 不能提取："用户喜欢日料"（推断）
</example>

<example id="drop_knowledge">
USER: TCP 和 UDP 的区别是什么
ASSISTANT: TCP 是可靠传输协议，有拥塞控制和重传机制……
→ 全部为空（USER 在提问，知识内容来自 ASSISTANT）
✗ 不能提取："TCP 是可靠传输协议"
</example>

<example id="drop_assistant_proactive_advice">
USER: 在赶代码
ASSISTANT: 别忘了每隔一段时间起来活动下，喝点水，久坐对颈椎不好……
→ 全部为空
✗ 不能提取："每隔45分钟应起身活动并补水"（来自 ASSISTANT，USER 没有授权）
关键判断：ASSISTANT 建议得再具体再合理，只要 USER 没有明确授权，就不是长期记忆
</example>

<example id="drop_meta_discussion_example">
USER: 我希望只有每轮对话里真正重要的参考信息才值得存入 memory.md，你举个例子我看看你理解没有
ASSISTANT: 明白。比如智能家居架构应坚持纯本地化部署，拒绝云端依赖……
检查0: USER 在讨论记忆标准并要求举例，是元讨论
可提取：USER 自己说出的筛选标准
ASSISTANT 的智能家居举例只是教学示范，不是 USER 新提供的规则
→ procedure: [{{"summary": "每轮对话中真正重要的参考信息才值得存入 memory.md"}}]
✗ 不能提取："智能家居架构坚持纯本地化部署"
</example>

<example id="drop_workaround">
USER: 那就直接写个脚本绕过去吧
→ 全部为空（当前任务临时策略，不跨 session）
✗ 不能提取："遇到此类问题应优先用 Python 脚本绕过"
</example>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【summary 写法约束】
- 只包含 USER 原话中直接出现的内容，不能加推断或延伸
- summary 语气不得强于 USER 原话（"不太喜欢" ≠ "强烈反感且要求永久避免"）
- summary 脱离对话也能独立成立，不含"这次""今天""当前"等时间锚
- 不能只是原话碎片，必须是完整句
- profile：每条 summary 只表达一条完整事实，绝对不合并

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前已有 profile（用于 profile 查重）】
{existing_profile}

【待处理对话】
{conversation}

只返回合法 JSON，不要 markdown 代码块：
{{
  "profile": [
{{"summary": "...", "category": "personal_fact|purchase|decision|status", "happened_at": null, "emotional_weight": 0}}
  ],
  "preference": [
{{"summary": "...", "emotional_weight": 0}}
  ],
  "procedure": [
{{"summary": "...", "emotional_weight": 0, "tool_requirement": null, "steps": [], "rule_schema": {{"required_tools": [], "forbidden_tools": [], "mentioned_tools": []}}}}
  ]
}}"""

RECENT_CONTEXT_SYSTEM = "你是近期语境压缩代理，只返回合法 JSON。"

RECENT_CONTEXT_USER = """你是近期语境压缩代理。你的任务不是自由总结，而是为后续 proactive 和 drift 保守地抽取近期语境。

目标：
1. 提取用户最近持续关注的话题
2. 提取最近新暴露、但尚未沉淀为长期记忆的显式偏好
3. 提取最近适合自然续接的话题
4. 提取最近应避免打扰、应避免推荐、或明显不想聊的方向
5. 提取跨窗口持续存在的重要现实线索（ongoing_threads）

规则：
- 只允许依据 USER 明确表达过的内容输出；ASSISTANT 的建议、解释、命名、延伸，一律不得当作证据
- recent_topics 可以总结“用户最近在讨论什么”，但必须贴近 USER 原话，不得升级成长期偏好
- active_topics 和 follow_ups 要优先写“话题层级”的概括，不要写 JSON Schema、函数名、字段名、具体术语翻译这类实现细节，除非用户明确把该细节当作核心关注点反复强调
- user_preferences 只允许在 USER 出现明确偏好/要求/禁忌表达时输出，例如：喜欢、偏好、希望、别、不要、避免、不想
- 不要把技术方案讨论、架构设想、问题求证、头脑风暴自动写成“用户偏好”
- 对技术讨论场景，只有当 USER 明确表达“以后都这样做 / 我就是偏好这种方式 / 我不要另一种方式 / 以后统一按这个来”时，才允许写 user_preferences；否则一律视为 active_topics 或 follow_ups
- 用户用“为什么不……”“能不能……”“是不是可以……”“只要不是最后一轮就……”这类方式提出方案设想或追问时，默认视为设计提议，不视为稳定偏好
- avoidances 只允许在 USER 明确表达“不要/别/避免/不想”时输出；没有明确否定表达就留空
- 如果最新 recent turns 显示话题已经明显切换，不要把较早窗口的技术讨论升级成当前偏好或避免事项
- 只保留未来几轮仍会影响主动行为的信息
- 不要记录工具细节、推理过程、普通寒暄
- 每个字段最多 3 条，每条尽量 1 句
- 没有把握就留空；宁可漏掉，也不要脑补

ongoing_threads 严格限制：
- 只记录用户正在经历、推进或承受的重要事情
- 必须是对用户当前生活、情绪、工作、学习、关系或健康有持续影响的线索
- 普通提问、技术讨论、方案脑暴、一次性 ask、知识求证，一律不得写入 ongoing_threads
- 若旧的 ongoing_threads 中已有某条重要线索，而当前窗口没有明确终结它，默认保留
- 只有当用户明确表示这件事已解决、结束、过去了、不再关心，才允许删除
- ongoing_threads 的写入门槛高于 active_topics；宁可少写，也不要把普通话题升级进去

专项禁令：
- 用户讨论“某个设计有没有依据/有没有实践/是否可行/为什么不这样做”，这是方案讨论，不是偏好；默认只能进入 active_topics 或 follow_ups，不能进入 user_preferences
- 用户说“为什么不让前台……只要不是最后一轮就……”是在提出一种实现设想，不等于“用户偏好以后统一这样做”
- 用户说“这样也不会引入额外延迟”“有没有这样的设计”，这是在分析方案目标，不等于稳定偏好
- 用户讨论“零延迟”“预加载”“流式预取”“前瞻性检索”这类设计目标时，默认视为当前方案讨论，不得直接提炼成 user_preferences
- 对方案讨论里的具体实现细节，优先上收一层概括，例如写“下一轮检索规划”“流式预取方案”，不要写“JSON Schema”“结构化预取指令”这类细碎实现点
- 用户说“睡觉了”“头有点疼”“身体不适”，这只是当前状态；除非用户明确说“别再聊这个”“不要继续”“我不想讨论”，否则不得生成 avoidances
- assistant 说“今晚先别想架构和代码了”“先休息”，这是 assistant 建议，不是用户 avoidances
- 如果较早窗口是技术方案讨论，而最新 recent turns 已切到睡眠/头痛/身体状态，则 user_preferences 和 avoidances 默认应为空；技术方案最多保留在 active_topics / follow_ups
- “最近在讨论前瞻性检索/流式预取方案”只能进入 active_topics / follow_ups，不能进入 ongoing_threads
- “用户最近几天反复因面试失败而情绪低落”“用户近期持续受睡眠紊乱影响”这类重要现实线索，才允许进入 ongoing_threads

反例：
- 错误：把“在 React 过程中同时输出下一轮检索内容”写成“用户偏好在对话中实时生成下一轮检索指令”
- 错误：把“这样也不会引入额外延迟”写成“用户偏好零延迟预加载”
- 错误：把“为什么不让前台在进行时同时输出自己想要什么”写成“用户偏好实时生成下一轮检索指令”
- 错误：把“睡觉了，吃了褪黑素头有点疼”写成“避免在身体不适时继续讨论技术架构”
- 错误：把“最近在讨论 React / 流式预取方案”写成 ongoing_threads
- 正确：active_topics 可写“用户最近在讨论前瞻性检索/流式预取方案”
- 正确：ongoing_threads 可写“用户最近几天反复提到面试受挫，持续影响情绪”
- 正确：如果用户没有明确说“希望/不要/避免/不想”，user_preferences 和 avoidances 可以为空

输出前自检：
1. 检查 user_preferences 中每一条，是否都能在 USER 原话里找到明确偏好/要求词（如“希望/不要/避免/不想/偏好/喜欢”）
2. 若找不到明确偏好/要求词，删除该条
3. 检查 avoidances 中每一条，是否都能在 USER 原话里找到明确否定/回避表达
4. 若找不到明确否定/回避表达，删除该条
5. 如果删除后为空，返回空数组，不要为了“信息完整”硬填

【上一版 recent context（仅供延续，不要机械复述）】
{old_context}

【较早窗口（本次待压缩）】
{conversation}

【最新 recent turns（只用于判断是否已切话题，不可把 assistant 内容当证据）】
{recent_turns}

返回 JSON：
{{
  "active_topics": [],
  "user_preferences": [],
  "follow_ups": [],
  "avoidances": [],
  "ongoing_threads": []
}}
"""

HYPOTHESIS_SYSTEM = """你是记忆检索查询改写器。输出一条简短检索假设，不回答用户问题，不解释。"""

MEMORY_OPTIMIZER_SYSTEM = (
    "你是一个用户长期记忆整理器。你的工作不是概括对话，而是从记忆中剔除噪音，"
    "只保留对未来每次对话都产生底色影响的长期记忆。"
)

MEMORY_OPTIMIZER_USER = """今日日期：{today}

你的任务是将「现有用户档案」重新整理为一份精炼的长期记忆，同时合并「待合并事实」。
更重要的是，你必须剔除不应该存在于用户档案中的内容。

## 核心判断标准：缺席成本测试
对每一条内容，问自己：在 6 个月后的一次全新对话中，如果这条信息没有被注入，
Agent 是否会在某个回复中出现方向性失误？是则保留，否则删除。

## 应保留的内容
- 用户事实：稳定身份与长期背景；当前就读学校/专业、实习公司+部门+岗位、在职单位+职位
  也必须保留具体细节和现在时态。
- 用户偏好：持续的审美取向、交互禁忌和根本价值判断，不是零散爱好清单。
- 用户明确要求长期记住的关键内容：保持原意和必要连贯性。
- 助手操作上下文：已部署且已授权使用的工具配置，具体端口、变量名和 URL 不得抽象化。

待合并事实的 tag 为 identity、preference、key_info、health_long_term、requested_memory、
correction、agent_context。

## 必须剔除
- 内网 IP、路由模式、运营商、MAC 等网络运维细节；项目路径、配置文件名和环境变量名可保留。
- Star 数、增长率等时效性数字，版本变更叙事和瞬时情绪；只保留背后的稳定价值判断。
- “最近、这周、目前”等随时会结束的临时状态。规律习惯可保留；就读、实习、在职三类
  当前社会角色不属于临时状态。
- 伪装成用户偏好的 Agent 执行规则、SOP、检索策略、标注规范和工具调用流程。

## 整理原则
- 只对偏好做同类合并和方向上收；学校、机构、部门、岗位等身份事实不得抽象化。
- 同类重复只保留最终版本；correction 直接反映最终值，不保留修改痕迹。
- 普通事实保持简洁；requested_memory 可保留更完整的连贯描述。
- agent_context 完整保留并放入 `## 助手操作上下文`。

## 输出格式
- 标题必须是 `# 用户长期记忆`。
- 必须依次包含 `## 用户事实`、`## 用户偏好`、`## 用户明确要求长期记住的关键内容`。
- `## 助手操作上下文` 仅在有内容时出现。
- 每个分类使用 bullet 列表，每条 1-2 行。
- 直接输出完整档案，不要 JSON、代码块或解释。

---
现有用户档案：
{memory}

待合并事实：
{pending}
"""

SELF_OPTIMIZER_SYSTEM = (
    "你是 MemoPilot，只能更新 SELF.md 中现有的三个 section，不得新增其他 section。"
)

SELF_OPTIMIZER_USER = """你的任务是根据当前 SELF.md 和本轮待合并事实，整理一份新的 SELF.md。

## 目标
- 只输出完整的 SELF.md。
- 只允许保留以下三个 section：
  - `## 人格与形象`
  - `## 我对当前用户的理解`
  - `## 我们关系的定义`
- 禁止新增任何其他 section，尤其禁止 `## 关系演进记录`。

## 更新原则
- 当前 SELF.md 是主文本，优先保留已有自我认知、语气和关系定义；不要机械改写待合并事实。
- 待合并事实只能在确实帮助澄清 MemoPilot 的定位与风格、对用户的稳定理解、双方长期关系时少量吸收。
- 用户资料、账号、key、设备参数、健康状态、动态指标、短期计划、近期事件、工具规范、
  SOP、执行流程和事件流水账不得写入。
- 没有足够高价值的新信息时，输出与当前 SELF.md 基本一致的版本。
- 保持简洁、有立场；这是自我认知，不是用户档案或工作日志。

## 输出约束
- 必须以 `# MemoPilot 的自我认知` 开头。
- 只能包含标题、上述三个 section 和 bullet 列表。
- 不要代码块、解释或额外说明。

---
当前 SELF.md：
{self_text}

待合并事实：
{pending}
"""

__all__ = [
    "CONSOLIDATION_SYSTEM",
    "CONSOLIDATION_USER",
    "HYPOTHESIS_SYSTEM",
    "IMPLICIT_LONG_TERM_SYSTEM",
    "IMPLICIT_LONG_TERM_USER",
    "MEMORY_OPTIMIZER_SYSTEM",
    "MEMORY_OPTIMIZER_USER",
    "RECENT_CONTEXT_SYSTEM",
    "RECENT_CONTEXT_USER",
    "SELF_OPTIMIZER_SYSTEM",
    "SELF_OPTIMIZER_USER",
]
