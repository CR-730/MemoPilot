# MemoPilot 独立实现设计

> **2026-07-28 架构覆盖说明：** 本设计中关于三进程、AgentJob／Run／Step、Inbox／Outbox、Outbound Effect、MessageBus 被动旁路和独立 BackgroundTaskLoop 的描述已被删除导向重构取代。当前批准并实现的主链路是：
>
> `Channel / ApplicationScheduler / Turn 后处理 → RedisTaskQueue(P0～P3) → AgentLoop → TaskDispatcher → 领域入口 → 必要时 MessagePushTool → Channel → ACK`
>
> SQLite 只保存会话、最终消息、定时执行、主动决策和记忆等业务事实；Redis 保存优先级、Pending、Lease、Fencing 协调与抢占信号。发送成功后才 ACK；发送失败保留 Pending。`activity_version` 继续作为主动发送的陈旧结果屏障。下文旧章节仅保留为历史方案，不再作为当前实现合同。

## 1. 背景与目标

MemoPilot 是一个单所有者、自托管的主动式个人 Agent，面向 AI 应用／Agent 开发岗位作品集。项目重点展示可解释的 Agent Runtime、主动任务、分层记忆、插件扩展、MCP、Skills 和 AI 后端工程能力。

第一版只支持飞书机器人私聊，不做多用户、多渠道和 Dashboard。FastAPI 只提供只读 Inspector 与健康检查接口。

## 2. 技术与产品边界

- Python 实现核心系统。
- 模型层定义统一 OpenAI-compatible Provider，第一版使用 DeepSeek。
- Embedding 使用独立 Provider 配置。
- MCP 仅支持 stdio，外部 MCP Server 作为独立依赖部署。
- 不使用 LangChain、LangGraph 等框架编排核心 Agent Loop。
- 不引入 PostgreSQL；长期数据由 Markdown 和 SQLite 保存。
- Redis 只负责任务队列、优先级、短期状态、锁、幂等与实时事件。
- Docker Compose 启动 app、worker、scheduler、redis。

## 3. 总体架构

### 3.1 App

App 负责接收飞书私聊事件和提供 Health、Readiness 与第 16.3 节定义的只读 Inspector API，不执行模型推理。MessageBus 后的 InboundBridge 在 `operational.db` 同一事务中写入飞书 inbox、递增权威 `activity_version`、创建 P0 AgentJob 和 OutboxEvent；提交后由 App 内的 OutboxDispatcher 尝试发布到 Redis。飞书事件 ID 或 message ID 在 inbox 中具有唯一约束，进程内 MessageDeduper 只承担快速去重。

### 3.2 Worker

Worker 是 `operational.db` 运行与会话数据、`memory2.db` 和 `wake.db` 的主要处理者。它执行被动回复、主动判断、用户定时任务、Drift、Consolidation 和 Memory Optimizer，并负责调用模型、工具、MCP 和飞书发送接口。任何需要派生异步任务的持久化提交都同时写入 operational outbox，不直接假设 Redis 发布一定成功。

### 3.3 Scheduler

Scheduler 统一管理主动轮询、用户定时任务和系统维护任务。它在 `operational.db` 同一事务中创建到期执行实例、推进下一次触发状态并写入 OutboxEvent，但不直接运行 Agent，也不把 SQLite 更新和 Redis XADD 当作一个不可分割操作。

### 3.3.1 三进程边界与取舍

原型面向本地单用户运行，使用单进程 asyncio Queue 串联 Channel、AgentLoop、Scheduler 和主动任务；这种方案依赖少、开发和部署简单，符合原型目标。MemoPilot 保留其单所有者范围，但为了让入口接收、时间触发和不稳定的模型/工具执行具有独立故障边界，第一版物理拆分为 App、Worker、Scheduler 三个进程：App 保持飞书入口可响应，Scheduler 只生成确定性的到期实例，Worker 承担耗时推理和外部副作用。任一进程重启时，另外两者不需要随之退出，未完成事实仍由 SQLite 恢复。

该拆分不是为了宣称当前具有高并发或集群规模。第一版默认且演示时均只运行一个 App、一个 Scheduler 和一个 Worker 副本；Redis Consumer Group、Lease 与 Fencing 主要解决跨进程派发、优先级、崩溃接管和新旧 Worker 短暂重叠，不把多 Worker 吞吐扩容列为交付目标。三进程同时带来 Redis 双写一致性、Outbox、恢复扫描和运维成本，只有设计书列出的崩溃窗口测试通过后才视为收益成立；若这些可靠性证据无法在第一版完成，应回退到原型式单进程，而不是保留无法解释的分布式外壳。

### 3.4 Redis

Redis 保存不同优先级的待执行任务、会话 lease、用户活动版本镜像、中断标记和运行时状态。Redis 不是事实源：`operational.db` 保存 inbox、AgentJob、调度实例和权威活动版本。OutboxDispatcher 负责至少一次发布；启动 Reconciler 扫描仍为 queued 且没有终态 Run 的 Job 并重新生成发布记录，因此 Redis 数据丢失后可以恢复可安全重投的任务。

### 3.5 持久化

Markdown 保存可阅读、可迁移的长期记忆。SQLite 收敛为 `operational.db`、`memory2.db` 和 `wake.db` 三个领域库：App、Worker、Scheduler 只在各自职责内短事务写 operational 库，Worker 写记忆和 Wake，Inspector API 只读。跨库只保存稳定 ID，不声明 SQLite 无法提供的跨库外键或原子事务。

## 4. Agent Runtime

Agent Runtime 采用两层结构：外层是生命周期 Phase Pipeline，内层是 ReAct + Function Calling 循环。

生命周期包含 BeforeTurn、BeforeReasoning、PromptRender、BeforeStep、AfterStep、AfterReasoning 和 AfterTurn。每个 Phase 由多个模块组成；模块声明唯一 `slot`，使用 `requires` 声明依赖，使用 `produces` 声明产物。启动时检查重复 slot、缺失依赖和循环依赖，并按依赖进行拓扑排序。

Reasoner 在每一步执行 BeforeStep、调用模型、处理 Function Calling、执行工具、追加工具结果并执行 AfterStep。模型不再返回工具调用时结束；达到迭代上限时生成基于已有结果的阶段性总结。

## 5. 统一任务模型

飞书消息、主动检查、用户定时任务、系统维护和 Drift 均表示为 `AgentJob`。任务至少包含 `job_id`、`job_type`、`priority`、`session_key`、`payload`、`idempotency_key`、`created_at`、`retry_count` 和 `activity_version`。

任务优先级为：

1. P0：用户飞书私聊和取消指令。
2. P1：用户明确创建的提醒和定时任务。
3. P2：Agent 主动内容检查。
4. P3：不占用用户回复路径的记忆维护；Drift 由空闲的 P2 主动 Job 同轮执行。

每个任务执行时创建 `AgentRun`，状态在 queued、running、succeeded、skipped、cancelled、failed、needs_review 之间转换。每次模型调用、工具调用、记忆检索和最终决策生成可查询的 `RunStep`。

## 6. 会话串行化与主动任务中断

同一 `session_key` 同时只允许一个 Turn 执行。App 收到用户消息后先在 `operational.db` 事务中递增权威 `activity_version`，再通过 Outbox/Redis 镜像发布 P0 Job 和中断信号。Redis 不可用时主动任务不得继续提交外部副作用。

主动任务在开始执行、每次模型调用前、每次工具调用前和发送消息前检查中断状态、权威活动版本以及会话 fencing token。最终推送前必须确认 lease owner、fencing epoch 和启动时活动版本仍有效。检查失败后当前执行者立即失权，只能记录 cancelled/superseded，不得写入后续 RunStep、ACK 或发送消息。

用户明确创建的定时任务不会因聊天而丢弃，只延迟到当前 Turn 完成后执行。普通主动检查在会话繁忙时不创建 Job；极小竞态窗口内已经启动的主动 Job 由 `activity_version` 使其失权并取消，等待下一次固定 Tick，不重新排队。Drift 被用户消息立即取消；记忆维护任务暂停后重试。

## 7. 已确认的持久化与扩展范围

- 完整保留短期会话、Markdown 长期记忆、SQLite + sqlite-vec 向量记忆三层结构。
- 内置工具包括记忆、任务调度、MCP 管理、飞书推送、文件读写和受限 Shell。
- 插件可以动态注册工具、注入提示词、插入 Phase 模块并拦截工具调用。
- 第一版通过 FastAPI Swagger 和结构化 API 展示运行轨迹，不开发独立前端。

## 8. 三层记忆系统

### 8.1 短期记忆

SQLite Session 保存完整消息，当前 Turn 使用滑动窗口控制上下文长度。`RECENT_CONTEXT.md` 保存 Compression、Ongoing Threads 和 Recent Turns，用于描述近期对话状态。

### 8.2 Markdown 长期记忆

- `MEMORY.md` 保存稳定身份、偏好和长期事实。
- `SELF.md` 保存 Agent 自我认知及与用户的关系。
- `HISTORY.md` 只追加时间线事件，不全文注入 Prompt。
- `PENDING.md` 保存等待 Optimizer 处理的长期事实候选。
- `RECENT_CONTEXT.md` 保存近期摘要、持续话题和最近对话预览。
- `journal/YYYY-MM-DD.md` 按日期保存事件。

`MEMORY.md`、`SELF.md` 和 `RECENT_CONTEXT.md` 每轮自动注入。Optimizer 低频批量归档 PENDING，避免高频修改稳定 Prompt，并使用 Snapshot、Commit、Rollback 两阶段提交。

### 8.3 SQLite 向量记忆

`memory2.db` 使用 sqlite-vec 保存 event、procedure、preference、profile。sqlite-vec 不可用时降级为余弦全表扫描。条目保存 source_ref、content_hash、reinforcement、emotional_weight、happened_at 和 status；冲突旧条目标记为 superseded，不直接删除。

## 9. 记忆检索

自动预检索与 `recall_memory` 复用同一个 Memory Engine 和 Retriever，但入口和输出合同不同。

### 9.1 自动预检索

每次普通用户 Turn 前自动执行，固定使用 `intent=context` 和原始用户消息。当前生产基线不装配 QueryRewriter，也不启用可选 HyDEEnhancer。向量 Lane 与关键词 Lane 的结果经 RRF 融合后，按类型阈值、相对分差、分段条数、行长和总字符预算生成文本块，并在首次模型推理前注入 Prompt。

### 9.2 recall_memory

`recall_memory` 由 Agent 在 ReAct 循环中按需调用，默认 `intent=answer`，支持 answer、timeline、interest、context、procedure、memory_kind、time_filter 和 limit。

- answer 使用原始 Query，并发生成 event/general 两个 HyDE 风格假设 Query。
- timeline 按时间范围直接查询 event。
- interest 只查询 preference 和 profile。
- procedure 只查询 procedure 和 preference。
- context 复用上下文检索路径。

工具返回结构化 JSON、分数、证据、source_ref 和引用要求，不直接注入系统 Prompt。Agent 使用任何记忆条目后必须在最终回复中引用相应 ID。

### 9.3 底层融合

原始 Query 和辅助 Query 分别进入向量 Lane；只有原始 Query 进入关键词 Lane。向量分数融合语义相似度与 Hotness。Hotness 由 reinforcement 与时间衰减相乘，emotional_weight 延长有效半衰期。向量与关键词结果使用 `1/(60+vector_rank) + 0.5/(60+keyword_rank)` 进行 RRF 融合。

## 10. 异步记忆写入

回复完成时在 `operational.db` 同一事务中提交消息、Run 终态、`TurnCommitted` 领域事件和 Consolidation AgentJob/OutboxEvent，不等待记忆处理。每个 Session 使用独立 lease 串行执行 Consolidation，并根据 `last_consolidated`、`keep_count` 与最小新消息数选择旧消息窗口。

Consolidation 使用稳定 `consolidation_id = hash(session_key, message_range)`。第一次模型调用提取 `history_entries` 和 `pending_items`，独立模型调用生成 RECENT_CONTEXT Compression；模型产物和目标文件 hash 先写入 `consolidation_manifests`。每个 Markdown 目标使用含 consolidation_id 的幂等标记、临时文件和同目录原子 replace；恢复时按 manifest 校验已完成目标并只续写缺失项。所有目标 hash 验证通过后，才在 operational 事务中把 manifest 标为 committed，并写入 `ConsolidationCommitted` OutboxEvent。

向量层订阅 `ConsolidationCommitted`，将 History Event 写入 event 记忆，并从同一对话窗口隐式提取 profile、preference 和 procedure。写入使用 source_ref、content_hash、向量预筛、LLM 去重决策、reinforcement 和 supersede 保证幂等与可纠正性。Post-response Worker 继续处理显式否定和纠正信号。

Redis 只提供至少一次异步任务承载；事件名称、处理顺序、窗口选择与记忆语义遵守本设计中的统一合同。`ConsolidationCommitted` 的消费者以 consolidation_id/source_ref 幂等，重复投递不得重复生成向量记忆。

## 11. 插件与工具扩展

插件通过受控 PluginContext 注册 PhaseModule、`@tool` 工具、`@on_tool_pre` 前置钩子和 Event Handler。Prompt 注入由参与 PromptRender 的 PhaseModule 完成。PhaseModule 使用 slot、requires、produces 参与拓扑排序；依赖图无效属于启动期配置错误。

内置工具、插件工具和 MCP stdio 工具统一进入 ToolRegistry。注册时验证名称冲突、Function Calling JSON Schema、工具来源、风险信息及允许使用的运行场景。PromptBlock 声明 block_id、priority、scope 和 max_chars，由 Prompt Renderer 统一应用预算。

Skills 是 Markdown 操作流程，不直接获得 Python 权限。Skill 中的所有动作仍必须通过已注册工具执行；Drift 只运行明确标记为允许后台执行的 Skill。

## 12. 工具 Hook 与 Shell 安全

工具调用依次经过参数校验、BeforeToolCall Hooks、Shell 内建校验、实际执行、AfterToolCall Hooks、结果截断和审计。Before Hook 可以拒绝调用或返回 updated_input 改写参数；原始参数、最终参数和 Hook Trace 均写入 RunStep。

`shell_restore` 解析以可选 sudo、env 或环境变量赋值为前缀的独立 rm 命令，将其改写为 `mv -- <targets> <restore_dir>`。还原目录自动创建且可配置。复杂管道或串联命令不进行猜测式改写，应由策略拒绝或要求拆分。

`shell_safety` 拒绝交互式编辑器、可能等待密码的 sudo、缺少非交互参数的软件包写操作、systemctl edit 和 crontab -e。

ShellTool 自身继续实施命令黑名单、网络目标与上传限制、受限任务目录、超时、后台任务管理、进程树终止和输出截断。第一版不增加飞书人工审批。

## 13. 主动唤醒与事件流

主动业务链以旧原型 `proactive_v2` 的实际 AgentTick 为准，外层保留 MemoPilot 的固定 Tick、Redis 优先队列、SQLite Reservoir、Lease/Fencing、`activity_version` 抢占和 Effect/Outbox。系统不使用 Content Hazard、Energy、AnyAction 或动态 Tick；检查周期固定为 30 分钟，模型只负责语义判断，不负责修改下一次调度时间。

### 13.1 调度与事件分类

Scheduler 可用短轮询扫描用户定时任务，但每 30 分钟才尝试创建一次 P2 `proactive.tick`。创建 P2 前，在 `operational.db` 的同一个写事务中检查目标会话是否存在 `queued` 或 `running` Job；只要会话被任何任务占用，本轮就不创建 Job 和 Outbox，等待下一个 30 分钟 Tick。P2 真正执行时先检查 Lease、Fencing 和 `activity_version`，通过后才重放 ACK、拉取 MCP Source 和调用模型。

Worker 通过配置驱动的 MCP Gateway 拉取 `alert`、`context` 和 `content`，先写入 SQLite Reservoir，再形成单 Tick 静态快照。路由严格互斥为 `Alert > Content > Context > Drift`：只把当前最高优先级的一类数据交给 AgentTick，不能让同一轮 Prompt 同时处理多类路径。Content 最多取 5 条并并发预取正文，但 Prompt 默认只展示元数据，正文由模型按需读取。

- `alert` 是高优先级快速路径。同 Tick 全部 Alert 由 AgentTick 合并成一条自然消息，evidence 必须包含全部 Alert ID，不执行 Content 分类。
- `content` 是主要主动发现路径。Agent 对每条候选分别调用 `recall_memory`、按需读取正文和外部验证，并显式执行 `mark_interesting` 或 `mark_not_interesting`。
- `context` 只在本轮没有 Alert 和 Content 时参与；每次以 30% 概率进入 AgentTick，未命中时仅更新本地 Context 快照。

Alert 不受每日次数、普通冷却和静默时段限制，但相邻两次确认发送至少间隔 30 分钟，冷却中的新 Alert 保留到下一轮合并。Content 与 Context 分别每天最多确认发送 3 次，共享 2 小时确认发送冷却，并只在配置时区的 08:00（含）至 23:00（不含）发送；窗口边界可配置。所有预算只统计 Effect 已确认且 Decision 已提交的真实发送，skip、failed 和 cancelled 不占额度。
- `context` 保留 Source 原始字段，只补充 `_source`、本地时间和 `awake_prob` 等展示信息；它只作为判断打扰时机的背景，或在 Wake Policy 明确放行且 Alert/Content 均无发送结果时作为末级 fallback，不再运行独立 transition LLM。

### 13.2 统一 AgentTick ReAct

三类 Source 复用同一个 AgentTick ReAct 实现，但每个 Tick 只选择一条业务路径。`message_push` 只暂存草稿，`finish_turn` 才提交终态；工具参数错误、抓取失败和普通工具异常都作为 Observation 交回模型，由模型决定修正、重试或退出。Content 若仍有漏分类项，Runtime 会恢复循环并要求补齐，避免跳过候选后直接结束。

AgentTick 的长期记忆、`RECENT_CONTEXT.md`、Workspace 主动规则与 Source 快照职责分开：记忆用于兴趣判断，Workspace 文件用于过滤规则，Context 用于打扰时机；只有 Alert/Content 及其验证结果可以成为外部事实来源。最终发送前执行来源级 delivery key 去重与旧原型式 LLM 语义去重。

### 13.3 ACK 与可靠消费

ACK 用于通知外部 MCP Source 某些事件已经处理，防止其在后续轮询中无限重复返回。消费使用本地事务性 Outbox 语义：先将 reservoir 事件标记为 consumed，并把 `(source_id, source_event_id, ack_operation_id)` 写入 `pending_acknowledgements`；随后调用 Source 的 ack tool。调用失败时保留 pending 记录，由后续 Tick 或重试任务使用同一 operation_id 再次发送；成功后才标记 acknowledged。

注册为 proactive source 的 ACK 工具必须声明并满足按 `source_id + source_event_id` 或 `ack_operation_id` 幂等；远端 ACK 成功、本地尚未标记便崩溃时允许安全重放。无法提供幂等 ACK 的 MCP 只能作为普通工具。Content 只有关联 outbound effect confirmed 后才消费 interesting 条目；明确丢弃项在 skip 提交后即可消费。发送成功后 cited、interesting-but-uncited、discarded 的 ACK TTL 分别为 168、24、720 小时；来源级或语义去重命中的候选按 post-guard 规则使用 24 小时 ACK。Alert 同样只在发送 confirmed 后整批消费并 ACK。Context 快照只在本地消费，不调用 Source ACK；Context fallback 发送失败时保留快照等待后续 Tick。

### 13.4 Drift

本轮没有 Alert、Content，且 Context 概率门未触发时，若距离上次 Drift 至少 3 小时，系统从 `background_allowed=true` 且依赖可用的 Skills 中选择任务，并在当前 P2 Job、当前 Lease 和 Fencing 下直接进入 Drift Runtime，不再创建第二个 P3 Job。Drift 复用既有 AgentRuntime、Skill 注入、共享工具、MCP 挂载、工具审计和抢占检查；是否调用一次 `message_push` 由 Drift ReAct 自己决定，不套用 Content/Context 的每日次数、两小时冷却和静默时段。

所有主动消息和 Drift 在模型调用前、工具调用前及最终副作用前重复检查中断标志。飞书发送前必须原子校验 `activity_version`，因此主动唤醒算法本身不承担与用户聊天并发协调的职责。

## 14. 用户定时任务

用户明确创建的定时任务与固定 30 分钟的 `proactive.tick` 分开建模。Agent 通过 `schedule`、`list_schedules` 和 `cancel_schedule` 三个工具管理任务；任务定义持久化在 SQLite，Scheduler 负责扫描到期定义、向 Redis 投递 P1 执行实例并推进下一次触发状态，但不运行 Agent。

### 14.1 触发与执行模式

定时任务支持三种触发方式：`at` 表示绝对时间，`after` 表示相对延迟，`every` 表示固定间隔或 cron 周期。`after` 必须基于用户消息的接收时间计算，而不是基于工具执行时间，避免模型推理和工具调用耗时造成提醒漂移。默认时区为 `Asia/Shanghai`，同时在每条任务中保存显式时区。

执行模式分为两种：

- `instant`：到期后直接发送预先确认的固定消息，不调用 LLM，适合喝水、会议等普通提醒。
- `agent`：到期后创建完整 Agent Turn，允许结合当前时间、记忆和工具获取实时信息，适合天气、新闻和周期总结。

第一版只有飞书私聊，因此目标 `session_key`、`open_id` 和 chat 标识必须从创建任务时的受信会话上下文取得，不作为可由模型自由填写的工具参数，防止跨会话发送。

### 14.2 持久化、投递与幂等

`operational.db.scheduled_tasks` 是任务定义和下一次触发时间的事实源。Scheduler 对到期任务使用 `task_id + scheduled_at` 作为 execution/idempotency key，并在同一 SQLite 事务中创建 execution、AgentJob、OutboxEvent 以及推进下一次触发时间。Redis 只承载发布后的执行副本；重复 XADD 或恢复重投由 execution 和 AgentJob 唯一键吸收。

一次性任务成功投递后进入 completed；周期任务按其时区计算下一次 `next_run_at`。若服务停止期间错过多个周期，第一版采用 coalesce 策略：只创建最近一次到期实例，并直接推进到未来的下一次触发时间，避免恢复后连续补发过期提醒。每次创建实例、跳过错过周期、执行成功或失败都写入审计记录。

### 14.3 会话协调

到期实例使用 P1 优先级，高于主动内容和 Drift，低于用户消息。用户正在聊天时任务不会丢弃或取消，而是在同一 `session_key` 的 P0 Turn 完成后继续执行。`instant` 发送和 `agent` 最终发送同样必须持有会话执行权并检查待处理用户消息，避免在回复中间插入提醒。

`agent` 模式完整记录模型调用、记忆检索、工具调用、最终消息和失败原因；`instant` 模式仍创建 `AgentRun` 与发送步骤，以便统一查询运行历史。取消任务只阻止尚未创建的新执行实例；已经入队但未开始的实例在执行前检查任务状态并安全跳过。

## 15. 失败恢复与重试

系统将“对话语义恢复”和“基础设施执行恢复”分开。工具或业务失败首先作为 ReAct 的观察结果交回 LLM，由 Agent 决定修正参数、选择替代工具、基于部分结果完成回答或自然语言说明失败；Redis 重投和任务重试只解决进程崩溃、网络瞬断等执行问题，不能盲目重复具有外部副作用的操作。

### 15.1 工具失败作为观察结果

ToolExecutor 捕获参数校验失败、MCP 异常、超时、Hook 拒绝、Shell 安全拒绝和工具自身异常，并返回统一工具观察。错误结果包含 `status`、`error_type`、面向模型的 `message`、`retryable`，同时保留最终参数及 Hook Trace 供审计。错误消息必须脱敏并截断，不向模型暴露密钥、完整环境变量或内部堆栈。通用工具层不维护副作用状态机。

该结果以标准 tool message 追加到当前 ReAct messages，随后正常进入下一轮 LLM。模型可以在安全前提下修正参数重试、改用其他工具、使用已有证据回答，或者明确说明哪些步骤没有完成。自动重试次数仍受 Agent 最大步数和重复调用检测限制；模型不得把 error 或 unknown 状态描述为成功。

### 15.2 不完整执行的自然语言收尾

达到最大 ReAct 步数、上下文预算不足、检测到工具调用循环、插件要求提前停止，或者已经取得部分结果但无法继续时，Runtime 再调用一次禁用工具的 LLM，请其总结已完成事项、未完成事项、失败原因和建议下一步。该收尾回复基于现有观察生成，不再次产生副作用。

只有 LLM Provider 持续不可用、关键持久化依赖不可用或进程正在强制终止等无法再次调用模型的情况，才允许使用简短固定兜底文案。固定兜底属于最后保障，不作为普通工具失败的默认响应。

### 15.3 任务投递与执行租约

Redis 按 P0 至 P3 使用独立 Stream，并通过 Consumer Group 提供至少一次投递。Worker 优先检查高优先级 Stream；任务进入可证明的终态后才 ACK Stream 消息。Worker 为运行中的任务维护带 owner_id 与 fencing_epoch 的 lease 和心跳；恢复进程只有在 Redis lease 已失效且 operational Run 心跳超时后才接管 Pending 消息。接管者必须取得更大的 fencing epoch，旧 owner 不得再提交业务状态或外部副作用。

LLM 限流、临时网络错误、只读 MCP、记忆检索、幂等 ACK、Consolidation 和 Embedding 等任务可以进行有限次数的指数退避重试。飞书文本发送使用预先持久化且稳定的 `feishu_uuid`，在官方一小时请求去重窗口内允许复用该 UUID 重试；成功后保存 message_id。飞书外发继续由专用 Outbound Effect 状态机处理；普通 Shell、MCP 等工具失败只作为 ReAct observation 返回，不由通用工具层自动重放，外部服务需要自行提供幂等键或查询凭证。

### 15.4 各类任务的失败边界

- 用户 Turn 的工具失败继续留在 ReAct 中，由 LLM 恢复或解释；Provider 完全不可用才返回固定兜底。
- `instant` 定时任务使用到期实例幂等键；明确未发送时可以重试，结果不明确时先核对飞书侧状态。
- `agent` 定时任务复用完整 ReAct 语义恢复，最终发送继续遵守副作用规则。
- 主动 content 判断失败时不 ACK，事件保留在 reservoir；alert 发送失败时不消费、不 ACK。
- Drift 失败后进入冷却期，避免持续消耗模型和工具。
- Consolidation 使用 `session_key + message_range` 幂等；Memory Optimizer 失败时回滚 Snapshot，不发布半成品。

超过基础设施重试上限的任务写入 SQLite 失败记录并可由 FastAPI 查询。第一版不开发独立死信管理界面。

## 16. 日志、Strategy Trace 与 Inspector API

第一版明确不开发 Dashboard：不实现前端页面、图表、Dashboard 插件宿主、动态 JS/CSS 资源、实时刷新和可视化 CRUD。系统只提供运行日志、策略 Trace 和少量只读 Inspector API，并通过 FastAPI `/docs` 查询和演示。

### 16.1 Diagnostic Log

运行日志使用固定字段单行格式。字段顺序为 `event`、`flow`、`phase`、`session`、`turn`、`tick`、`action`、`reason`、`duration_ms`、`counts`、`error_type`、`error_fp` 和 `note`。`session/flow/phase/turn/tick` 通过 ContextVar 在异步调用链传播，模块只补充当前事件特有字段。

日志面向 Docker stdout、开发调试和文本检索；换行与双引号需要清理，错误堆栈只在异常日志中输出。`session_key`、用户文本、工具参数和模型结果按敏感级别脱敏或截断，固定字段不得写入 API Key、Token 和完整环境变量。

### 16.2 Strategy Trace

需要长期审计的决策使用 JSONL envelope，固定包含 `trace_type`、`source`、`subject`、`ts` 和 `payload`。`subject` 包含 `kind` 与 `id`，用于关联 session、job、action 或全局配置；`payload` 由路由、主动唤醒、调度、记忆检索和插件加载等领域自行定义。

AgentRun/RunStep 继续承担 Redis 任务状态、幂等、失败恢复与内部执行审计，但不强制所有领域通过一套通用 `/api/runs` 暴露。消息详情中的 `tool_chain` 保存 ReAct 的工具轨迹；Recall、Wake 和 Schedule 分别由只读 Reader 组合自己的领域数据。

### 16.3 最小 Inspector API

第一版只提供以下接口：

- `GET /health/live`：报告 App 进程存活。
- `GET /health/ready`：检查 Redis、SQLite、配置和 Worker 心跳，不发起付费 LLM 请求。
- `GET /api/inspect/turns/{turn_id}`：返回 Phase、模型调用摘要、工具链和最终回复。
- `GET /api/inspect/recall/{turn_id}`：返回自动预检索与显式 `recall_memory` 的候选、分数、注入状态和引用。
- `GET /api/inspect/wake`：返回当前 Hazard、threshold、reservoir 数量、Pending ACK 和最近 Wake 摘要。
- `GET /api/inspect/wake/{wake_id}`：返回某次 Wake 的触发信息、候选内容、脱敏 LLM 输入与最终决定。
- `GET /api/inspect/schedules`：返回当前定时任务。
- `GET /api/inspect/schedules/{task_id}/executions`：返回某项任务的到期实例和执行结果。

列表型数据在上述详情接口内部使用 `page/page_size` 时，页码从 1 开始，`page_size` 上限为 200，返回 `items/total/page/page_size`。不存在的资源返回 404；依赖未启用时返回稳定的 `available=false` 状态，而不是 500。

Inspector API 默认只监听本机或 Docker 内网；对外开放时必须配置管理 Token。飞书长连接凭据与 Inspector 管理 Token 相互独立。

## 17. 持久化边界与 SQLite 拆分

为保证 inbox、会话提交、调度推进和 Redis 发布之间可以使用本地事务闭环，第一版不采用原先五库方案，而是收敛为三个领域库。workspace 包含 `operational.db`、`memory2.db`、`wake.db`、Markdown `memory/` 目录和 JSONL `observe/` 目录。

### 17.1 数据库职责

- `operational.db` 保存 sessions、messages、message extra/tool_chain、inbound_events、session_activity、AgentJob、AgentRun、RunStep、session_fences、scheduled_tasks、schedule_executions、outbox_events、outbound_effects、consolidation_manifests、失败记录和用户 Turn embedding。
- `memory2.db` 保存 event、profile、preference、procedure、sqlite-vec 向量索引、source_ref、content_hash、reinforcement 和 supersede 状态。
- `wake.db` 保存 alert/content reservoir、content embedding、Hazard、Wake observation、Pending ACK 和 Context 状态。

Markdown 长期记忆继续保存在 `memory/MEMORY.md`、`SELF.md`、`HISTORY.md`、`PENDING.md`、`RECENT_CONTEXT.md` 和 `journal/`；Strategy Trace 等追加型观察记录保存在 `observe/*.jsonl`。跨库引用只使用稳定字符串 ID，并由领域幂等约束维护，不声明跨库外键。

### 17.2 写入所有权与并发

InboundBridge、Worker 和 Scheduler 都可以在各自职责内短事务写 `operational.db`：InboundBridge 只写 inbox/activity/job/outbox，Scheduler 只写 schedule execution/job/outbox，Worker 写 session/run/step/effect/consolidation。Worker 写 `memory2.db` 与 `wake.db`；Inspector Reader 只读。

所有 SQLite 连接启用 WAL、foreign keys 和 5 秒 busy timeout，并对 `SQLITE_BUSY` 做有界退避。容器共享同一个本地 workspace volume；第一版不支持网络文件系统上的 SQLite，也不支持多个主机同时挂载数据库。若写竞争在集成测试中超过阈值，再把 OutboxDispatcher 拆为独立进程，而不是提前引入 PostgreSQL。

### 17.3 Schema 迁移

每个数据库使用 `PRAGMA user_version` 标记 Schema 版本，并提供按版本顺序执行的显式迁移。启动时先备份迁移目标数据库，再在事务中升级；版本高于当前程序支持范围时拒绝启动。不得仅依赖大量 `CREATE TABLE IF NOT EXISTS` 静默掩盖字段或约束差异。

## 18. 飞书 Channel

飞书接入以已有的 `FeishuChannel` 为行为基线，不重新设计为另一套 SDK 封装。根据第一版单用户自托管范围，只保留 `lark_oapi` 长连接，删除原模块的 webhook 接收模式；飞书私聊的消息解析、身份、资源和发送行为继续保持一致。

### 18.1 启动与长连接

Channel 启动时重建 `SessionIdentityIndex`，订阅 MessageBus 的飞书 outbound，并使用 `lark_oapi.ws.Client` 和 `EventDispatcherHandler` 注册 `im.message.receive_v1`。SDK Client 在独立 daemon thread 中运行；回调通过 JSON marshal 转为通用 payload，再使用 `asyncio.run_coroutine_threadsafe` 投递到主 asyncio loop 的 `handle_event`。停止时关闭 SDK 长连接并关闭共享 httpx client。

第一版不启动 Channel 自带 FastAPI/Uvicorn，不提供 URL verification，不处理 Verification Token、Encrypt Key 或 encrypted callback，也不保留 receive_mode 分支。项目中的 FastAPI 只负责 health 和 Inspector API。

### 18.2 事件处理与会话身份

Channel 只处理 `im.message.receive_v1`。事件使用 `event_id`，缺失时回退 `message_id`，通过容量 500 的 `MessageDeduper` 做进程内快速去重；进入 Redis 后继续由 AgentJob 幂等键承担跨进程防重。发送者身份依次选择 `open_id/user_id/union_id`，`allow_from` 可以匹配其中任意一种 ID。授权通过后将发送者与 `chat_id` 写入 `SessionIdentityIndex`，会话键继续使用 `<channel_name>:<chat_id>`。

消息标准化保留本地行为：text 提取正文，post 展平标题与多语言正文，image 生成 `[图片]` 并通过消息资源 API 下载到 workspace `uploads/`，file 提取文件名。InboundMessage metadata 保存 message_id、event_id、chat_type、message_type、open_id、user_id 和 union_id。

收到文本 `/stop` 时不进入普通 Agent Turn，而是调用 `InterruptController.request_interrupt`，使用相同飞书会话返回中断结果。其他消息发布到 MessageBus；App 中的 Redis bridge 负责更新 `activity_version` 并创建 P0 AgentJob，因此不修改 FeishuChannel 自身的总线接口。

### 18.3 消息发送与资源

Channel 继续通过 httpx 直接调用飞书 Open API，而不是把发送改写为另一套 SDK 抽象。tenant access token 按飞书返回过期时间缓存，并提前 300 秒失效。文本、图片和文件统一使用目标 `chat_id` 与 `receive_id_type=chat_id`。发送接口改为接收 Runtime 预先持久化的 `operation_id` 与稳定 `feishu_uuid`，并返回包含 `message_id` 的 `SendReceipt`；Channel 不再在重试时生成随机 UUID。飞书官方对相同 UUID 提供一小时请求去重，因此结果不明确时只能在该窗口内复用同一 UUID 重试，超过窗口转入 `needs_review`。图片先上传获取 image_key，文件先上传获取 file_key；`send_stream` 第一版与本地实现一致，退化为普通 `send`。

MessagePushTool 注册同一 Channel 的 text、stream_text、file 和 image sender。被动回复、主动推送和定时任务继续复用这一发送能力；主动与定时任务的会话抢占、fencing epoch 和 activity_version 校验发生在创建 `outbound_effect` 及提交发送之前。文本发送纳入完整幂等状态机；图片和文件的上传阶段不具备同等端到端幂等保证，第一版只要求被动会话手动演示，不允许 Wake 或定时任务自动重试上传。

### 18.4 配置与测试基线

配置字段缩减为 app_id、app_secret、allow_from 和 channel_name；Secret 允许通过环境变量占位符注入。验收测试复用本地行为基线，至少覆盖文本接收、重复事件、未授权发送者、`/stop`、文本发送、图片资源下载，以及 ws thread 到主 event loop 的桥接。原 webhook URL verification 测试不迁移。

该 Channel 的长连接核心实现及对应行为测试直接纳入新仓库；迁移时只做依赖接口适配和 webhook 删除。MessageBus、AttachmentStore、MessageDeduper、SessionIdentityIndex 和 Redis bridge 等公共依赖按本设计实现。

## 19. 可靠投递、租约与外部副作用

### 19.1 Transactional Inbox/Outbox

所有“提交数据库状态后还要投递 Redis”的入口均禁止直接先写库再 `XADD`。入站飞书消息在 `operational.db` 的一个事务中完成：插入唯一 `inbound_event`、递增会话 `activity_version`、创建唯一 `agent_job`、插入唯一 `outbox_event`；重复 `event_id` 或 `message_id` 返回已有结果。Scheduler、Turn Commit 和 Consolidation 也使用同一模式，把业务状态和待发布事件一起提交。

`OutboxDispatcher` 按 `next_attempt_at` 扫描 `pending` 事件，在 SQLite 事务中写入 `claim_owner/claim_until` 并转为 `publishing`，随后向相应优先级 Stream `XADD` 并把 job_id 加入 Redis `queued:job_ids` 镜像，成功再标记 `published`。过期的 publishing claim 可由其他 Dispatcher 重新认领。如果进程在 `XADD` 后、更新 SQLite 前崩溃，会发生重复投递，但消费者通过 `job_id`、`execution_id`、`consolidation_id` 或 `idempotency_key` 吸收。不得用 Redis 消息 ID 作为业务身份。

启动和每 60 秒运行一次 `QueueReconciler`：对照 SQLite 中仍处于 `queued`、没有终态 Run 的 Job 与 Redis `queued:job_ids` 镜像；镜像缺失时只对该 Job 已有的 published outbox 做 CAS `published -> pending`，清空 claim/published_at 并增加 recovery_count，不创建第二条相同 idempotency_key 的记录。Redis Stream 与镜像之间的崩溃窗口最多造成重复投递，不造成漏投。发布时禁止用 `MAXLEN` 盲目裁剪未完成消息，因为 Stream 与独立 Set 镜像可能因此分裂；Worker 把 Job 提交为终态后，才在一个 Redis 事务中执行 XACK、XDEL 并从镜像移除 job_id。若 Worker 在 SQLite 终态提交后、Redis 清理前崩溃，恢复器把该 Pending 认领为 cleanup，而不再次执行 Job。这样 Redis 空间随终态清理，同时即使 Redis 被清空，SQLite 中尚未完成的任务仍能恢复。

心跳过期的 running Job 只有在不存在不明确外部副作用时才可恢复。Reconciler 在 operational 事务中把 Job CAS 为 queued、把同一个逻辑 Run 标为 recovering，并关闭旧 `run_attempt`；消费者重新取得更大 fencing epoch 后，不插入第二个 Run，而是 CAS 更新现有 Run 的 owner/epoch、创建下一条 `run_attempt` 并从已持久化 Step/Message 安全检查点恢复。`runs.job_id UNIQUE` 保持不变。存在 unknown effect 时 Job/Run 直接进入 needs_review，不自动重排。

### 19.2 会话租约与 Fencing

Redis 可能被清空，因此不能作为 fencing epoch 的事实源。获取会话执行权分三步：Lua 先以 `SET NX PX 30000` 写入 `owner_id:provisional_token`；取得 provisional lease 后，在 `operational.db` 事务中将 `session_fences.current_epoch` 单调加一并写 owner/heartbeat；最后用 compare-and-set Lua 把 provisional 值替换为 `owner_id:epoch`。若最后一步失败，Worker 放弃执行，数据库 epoch 允许出现空洞但绝不回退。Worker 每 10 秒使用 compare-and-`PEXPIRE` 脚本续租，释放时使用 compare-and-`DEL`；旧 owner 永远不能续租或删除新 owner 的锁。Stream Pending 至少 idle 60 秒、Redis lease 已失效且 `operational.db` heartbeat 也过期后，才允许接管。

第一版不直接使用会在扫描时立即改变消息 owner 的 `XAUTOCLAIM`。因为“是否可接管”取决于每条消息所属会话的 Redis lease 和 SQLite heartbeat，恢复器用带游标分页的 `XPENDING` 找到超过 idle 阈值的候选，再读取业务身份、检查两项前置条件，最后用 `XCLAIM` 精确认领该消息；终态 Job 则直接标记为 cleanup，只清理 Pending，不恢复执行。该选择比一次 `XAUTOCLAIM` 多几个 Redis 往返，但避免先认领后才发现对应会话仍在运行，也避免固定首页候选阻塞后续可恢复消息；第一版只有一个 Worker，安全性收益高于这点吞吐成本。

Run、Step、Message、Consolidation 和 `outbound_effect` 的关键状态更新都携带 epoch，并以 `WHERE current_epoch = :epoch AND owner_id = :owner_id` 或等价事务校验提交。Job 由 `queued -> running` 使用原子 compare-and-set 认领，`runs.job_id UNIQUE` 阻止重复 Redis 消息并发创建两个 Run。Worker 在模型调用前、工具调用前、创建副作用前和最终提交前重新校验 lease、epoch；失权后立即停止写入。已经发出的网络请求不能撤销，继任者必须按 `outbound_effect` 状态核对，不能简单重发。

### 19.3 Outbound Effect 状态机

每次外发先创建唯一 `outbound_effect(operation_id, payload_hash, provider_uuid, expected_activity_version)`。真正调用飞书前，在同一个 `operational.db` 事务中校验 `session_fences.owner_id/epoch` 和 `session_activity.activity_version == expected_activity_version`，并把 effect CAS `pending -> sending`；该事务提交是主动发送的抢占线性化点。此前已提交的 P0 入站会使 CAS 失败并把 effect 置为 cancelled，此后才到达的 P0 无法撤销已经开始的网络请求。只有当前 fencing epoch 的 owner 可以继续推进状态；成功响应保存 `message_id` 并置 confirmed。连接超时、响应丢失等无法判断结果的情况进入 `unknown`：在首次请求后的一小时内，可使用同一 `provider_uuid` 和相同 payload 重试；一小时后仍不明确则进入 `needs_review`，Inspector 显示但不自动重发。

`provider_uuid` 使用固定 namespace 的 UUIDv5 从稳定 `operation_id` 确定性派生；同一 operation 不允许改变 payload_hash。该 Effect 合同只服务飞书外发，不扩展成所有工具共用的副作用状态机。普通工具只声明原型一致的 `risk`，MCP、Shell 和上传类操作由 ReAct 根据观察结果决定后续动作，系统不会在传输层盲目重放。

## 20. 最小数据契约

### 20.1 operational.db

以下是实现前必须固定的核心表及约束；审计详情可以增加字段，但不得删除身份、状态和唯一键：

| 表 | 核心字段与约束 |
|---|---|
| `inbound_events` | `event_id PK`、`message_id UNIQUE`、`session_key`、`payload_json`、`received_at` |
| `sessions` / `session_activity` | `session_key PK`、`activity_version`、`last_user_at`、`updated_at` |
| `agent_jobs` | `job_id PK`、`kind`、`priority`、`session_key`、`idempotency_key UNIQUE`、`state`、`activity_version`、时间戳 |
| `runs` / `run_attempts` / `steps` | `run_id PK`、`job_id UNIQUE`、owner/`fencing_epoch`、状态；Attempt 按 `(run_id, attempt_no)` 唯一并保存每次接管 owner/epoch/outcome；Step 按 `(run_id, step_index)` 唯一并保存 tool/observation |
| `session_fences` | `session_key PK`、`current_epoch`、`owner_id`、`heartbeat_at` |
| `outbox_events` | `outbox_id PK`、`event_type`、`aggregate_id`、`payload_json`、`idempotency_key UNIQUE`、`state`、`attempts`、`recovery_count`、`next_attempt_at`、`claim_owner`、`claim_until`、`published_at` |
| `outbound_effects` | `operation_id PK`、`run_id`、`session_key`、`payload_hash`、`provider_uuid UNIQUE`、`expected_activity_version`、`state`、owner/`fencing_epoch`、`message_id`、错误和时间戳 |
| `scheduled_tasks` | `task_id PK`、时间规则、执行模式、payload、`next_run_at`、`enabled`、`version` |
| `scheduled_executions` | `execution_id PK`、`task_id`、`scheduled_at`、`state`，`UNIQUE(task_id, scheduled_at)` |
| `consolidation_manifests` | `consolidation_id PK`、消息范围、目标 artifact/hash JSON、`state`、尝试次数、时间戳 |
| `messages` | `message_id PK`、`session_key`、role/content、`turn_id`、`created_at`；同 Turn 顺序唯一 |

状态值使用代码枚举并由数据库 `CHECK` 约束。Outbox 为 `pending|publishing|published|dead`；Job 为 `queued|running|succeeded|skipped|failed|cancelled|needs_review`；Run 额外允许 `recovering`；Effect 为 `pending|sending|confirmed|cancelled|unknown|needs_review`。所有时间统一保存 UTC ISO-8601，展示时才转换时区。

### 20.2 memory2.db 与 wake.db

`memory2.db` 至少包含 memory item、embedding、keyword index metadata、reinforcement/supersede 关系和 `source_ref UNIQUE`；向量维度写入 metadata，启动时必须与当前 Embedding Provider 匹配。`wake.db` 至少包含 source event reservoir、source cursor、pending acknowledgement、content score/hazard snapshot、wake decision 和 drift history；`UNIQUE(source_id, source_event_id)` 防止重复采集，`ack_operation_id UNIQUE` 保证 ACK 重放身份稳定。Wake decision 保存稳定 `effect_operation_id`：share/alert 只有对应 outbound effect 已 confirmed 后才在 wake.db 事务中 consume 并创建 pending ACK；skip 可在 skip 决策提交后直接 consume。若发送确认后、consume 前崩溃，恢复者通过 effect_operation_id 发现 confirmed，只补做 consume/ACK，不再次发送。

跨数据库只保存稳定 ID，不使用跨库外键。Consolidation manifest 负责 Markdown 多文件提交证明；只有目标文件 marker 与 hash 全部匹配后，才允许发布 `ConsolidationCommitted`。

## 21. 插件、MCP 与 Source Wire Contract

### 21.1 Python 插件运行时

“不开发插件宿主”仅指不开发 Dashboard 的前端插件宿主。Python Runtime 保留与参考原型一致的受信任本地插件机制：启动时扫描插件目录中的 `plugin.py`，创建插件实例并注入 PluginContext，按声明顺序直接注册 Event Handler、Tool、`@on_tool_pre` 和 PhaseModule，最后调用 `initialize()`。插件元数据只描述名称、版本、说明和作者，不承担依赖解析或权限沙箱职责。

`initialize()` 失败时回滚本轮已经注册的 Tool、ToolHook 和 PhaseModule，并记录诊断；EventBus handler 和模块导入产生的 Python 副作用不承诺事务性回滚，这与受信任本地代码边界一致。插件升级、启停或卸载需要重启，不做运行时热重载和进程隔离。

### 21.2 MCP stdio

MCP 配置每项包含 `server_id`、`command`、`args`、`env`、`cwd`、`enabled`、启动/调用/关闭超时、最大重启次数和模型可见结果字符预算。只允许数组形式 command/args，不经 shell 拼接；Secret 使用环境变量引用，Inspector 仅显示脱敏配置。进程异常退出最多重启 3 次，指数退避；单个 Server 不可用不能阻止核心 Agent 启动，但其工具调用返回结构化 observation。

第一版基于官方 MCP Python SDK v1，版本约束 `<2`，使用 `list_tools` 与 `call_tool`。MCP 工具映射到统一 ToolRegistry 时保留 `server_id/tool_name` 来源、输入 JSON Schema、超时和 `risk`；名称暴露为稳定别名，冲突时拒绝启动相应注册项。客户端内部保留完整结果供 Source 合同使用，进入模型上下文前按字符预算截断；图片、音频等二进制块只暴露类型、MIME 与大小摘要。

### 21.3 Wake Source

可注册为 Wake Source 的 MCP 必须额外提供以下逻辑契约（实际工具名由配置映射）：

- `fetch(cursor?, limit) -> {events, next_cursor}`；Event 必须含 `event_id`、`kind=alert|context|content`、`occurred_at`、`payload`，content 可含 `title/url/published_at/source`。
- `ack(event_id, operation_id) -> {acknowledged: bool}`；必须按 event_id 或 operation_id 幂等，多次调用返回相同成功语义。
- cursor 只有在 fetched events 已持久化到 reservoir 后才推进；fetch 超时不会推进。
- Source 必须声明 capability、超时、最大批量和 ACK 幂等保证；缺少保证时只能作为普通 MCP 工具。

## 22. 工程结构、配置与依赖

### 22.1 目录

```text
src/memopilot/
  app/              # 飞书长连接、FastAPI、Inbound/Outbox bridge
  runtime/          # Phase、ReAct、Job/Run、fencing、effects
  memory/           # retrieval、consolidation、optimizer
  proactive/        # source、reservoir、wake、drift、ack
  scheduling/       # task model、scheduler、coalesce
  plugins/          # manifest、loader、registries、hooks
  mcp/              # stdio client 与 source adapter
  channels/feishu/  # 飞书长连接与接口适配
  persistence/      # SQLite schema、migration、repositories、outbox
  observability/    # diagnostic log、strategy trace、Inspector readers
tests/{unit,integration,replay,e2e}/
config/
workspace/{data,memory,journal,uploads,restore,traces}/
docs/
```

### 22.2 配置和默认值

配置使用 Pydantic Settings；`load_settings()` 读取的 TOML 字段作为显式配置，优先于环境变量和 `.env`，未在 TOML 出现的字段再按环境变量、`.env`、字段默认值解析。生产 Secret 只从环境变量或本地配置引用，不提交仓库。关键默认值：主动 Tick 1800 秒、Context 概率门 30%、普通主动发送时段 08:00～23:00、Content/Context 各每日 3 次且共享 2 小时冷却、Alert 30 分钟防刷屏、Drift 最短间隔 3 小时、lease TTL 30 秒、heartbeat 10 秒、reclaim idle 60 秒、SQLite busy timeout 5 秒、MCP startup/call timeout 15/30 秒、LLM 可重试错误最多 2 次。Redis Stream 不配置发布时 `MAXLEN`，而是在任务终态 ACK 时精确删除对应消息。检索 top-k、Prompt/Step 预算和各重试上限均可配置，并在 Trace 中记录当次有效值。

启动时执行配置校验：DeepSeek Chat Provider 与独立 Embedding Provider 必须显式配置；embedding 维度必须匹配数据库；飞书 owner allowlist 不能为空；路径必须位于 workspace；MCP command 不允许 shell 字符串。`/health/live` 只检查进程，`/health/ready` 检查迁移、Redis、Provider 配置和飞书初始化状态。

### 22.3 依赖选择

以 Python 3.12 和 uv 管理依赖。第一版优先使用 FastAPI、Uvicorn、Pydantic Settings、redis-py、httpx、OpenAI Python client、MCP Python SDK v1、sqlite-vec、APScheduler 3.x、lark-oapi、pytest 与 pytest-asyncio；锁文件固定实际解析版本。MCP v2 和 APScheduler 4 均仍处于非稳定阶段，不纳入第一版。sqlite-vec 仍是 pre-1.0 组件，因此必须保留纯 SQLite 余弦扫描降级路径。

外部 MCP Server 不 vendoring；Docker 镜像和 Python 依赖在发布前执行许可证及漏洞扫描。依赖版本统一由 uv lockfile 固定，避免开发环境与部署环境产生隐式差异。

## 23. 两周交付边界

Must 范围直接对应简历：飞书文本私聊、外层 Phase + 内层 ReAct/Function Calling、统一工具错误 observation、三层记忆检索、异步 Consolidation 与向量写入、固定 Tick 的 content/alert/context Wake、允许列表内的 Drift Skill、`at/after/every` 定时任务、Python 插件、MCP stdio、Skills、Redis 优先级/抢占/租约、事务 Outbox、受限 Shell、日志/Trace/Inspector、Docker Compose 和上述自动化测试。

Stretch 范围不阻塞发布：主动图片/文件推送的端到端幂等、复杂 context 抑制与高级 Drift 调参、运行时插件热重载、MCP 进程池、完整媒体 E2E、更多 Inspector 聚合和跨主机部署。Must 中的 Memory Optimizer 只实现 PENDING 到 MEMORY/SELF 的可恢复批处理和快照提交，第一版不扩展额外启发式。

## 24. 测试与验收

第一版采用按风险分层的测试策略，不以单一覆盖率数字作为完成标准。核心目标是验证并发、幂等、记忆、主动决策和外部副作用边界，并让每条简历描述都有可运行、可查询的证据。

### 24.1 单元测试

单元测试至少覆盖 Phase 拓扑排序及错误诊断、ReAct 工具失败 observation、自动预检索与 `recall_memory` 分流、Consolidation 事件顺序与幂等、向量记忆 dedupe/reinforcement/supersede、简化 Hazard 的衰减与阈值、`at/after/every` 时间计算与周期 coalesce、Shell 改写与拒绝，以及飞书长连接的事件解析、去重、白名单、`/stop` 和资源收发。

时间、随机数、模型响应和外部工具必须可注入，涉及时间和 Hazard 的测试使用固定 Clock；不得用真实 sleep 等待。LLM 文本不按逐字匹配，测试结构化工具调用、状态转换和副作用。

### 24.2 集成与回放测试

集成测试使用真实 Redis 与临时 SQLite，使用 Fake LLM、Fake MCP 和 Fake Feishu API，验证 Redis Stream 优先级与 Pending 接管、同会话串行化、用户消息抢占主动发送、定时实例幂等、ACK Outbox 重试、Consolidation 异步链和 Worker 崩溃后的安全恢复。测试必须提供 failpoint，逐一覆盖“SQLite commit 前”“commit 后/XADD 前”“XADD 后/标记 published 前”“飞书收到请求后/本地收到响应前”“续租失败后工具仍返回”和“Consolidation 部分文件已替换”等窗口。

Wake 回放使用固定时钟、历史对话、alert/content/context 事件及确定的模型工具调用结果，检查 reservoir、Hazard、LLM 唤醒、share/skip、发送和 ACK 状态。回放比较决策和状态，不比较自然语言逐字内容。真实 DeepSeek、飞书和第三方 MCP 只用于手动端到端验收，避免自动测试产生费用和外部副作用。

### 24.3 必须通过的演示场景

发布第一版前必须能够现场演示：

1. 飞书私聊触发完整 ReAct + Function Calling，并通过 Inspector 查看工具链。
2. 工具失败进入下一轮 LLM，由模型自然解释、修正参数或选择替代工具。
3. 自动记忆预检索与显式 `recall_memory` 使用不同入口并返回可引用证据。
4. Turn 完成后异步执行 Consolidation、Markdown 写入和向量记忆写入。
5. Content reservoir 累积到 Hazard 阈值后进入 LLM share/skip 判断。
6. 用户消息抢占尚未提交的主动消息，且不会丢失用户定时任务。
7. `instant` 与 `agent` 两种定时任务按幂等语义执行。
8. Drift 只执行明确允许后台运行的 Skill。
9. 插件动态注册工具、PromptBlock、PhaseModule 和 ToolHook。
10. 外部 stdio MCP 注册、调用和错误 observation。
11. 独立 `rm` 被改写到恢复目录，危险或交互式 Shell 被拒绝。
12. FastAPI `/docs` 可以查询 Recall、Wake 和 Schedule 的 Inspector 数据。

