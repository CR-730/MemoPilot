# MemoPilot 工程学习笔记

> 用途：记录每个实施阶段必须真正理解、能够在面试中解释的工程知识。  
> 维护规则：阶段实现、验收和 Learning Opportunities 训练完成后更新对应章节；明确区分“已经实现”“设计目标”和“后续计划”。

## 阅读索引

| 阶段 | 主题 | 当前状态 |
|---|---|---|
| 阶段 0 | 配置、三库、SQLite、迁移、Embedding 身份 | 已实现、已学习 |
| 阶段 1 | Inbox/Outbox、Redis Streams、Lease、Fencing、故障恢复 | 已实现、已学习 |
| 阶段 2 | Phase DAG、ReAct、Provider、ToolRegistry、运行审计 | 已实现、已学习 |
| 阶段 3 | 飞书私聊、实时执行反馈、外部副作用、`/stop` 中断 | 已实现、已学习；真实飞书端到端已验收 |

---

# 阶段 0：工程基线与数据事实层

## 1. 配置系统

### 1.1 配置来源与优先级

MemoPilot 使用 Pydantic Settings。配置优先级从高到低为：

```text
Python 显式初始化参数
→ 操作系统环境变量
→ .env
→ config/default.yaml
→ Python 字段默认值
```

示例：

```text
YAML = 10
.env = 20
操作系统环境变量 = 30
MemoPilotSettings(llm_max_iterations=5)

最终值 = 5
```

显式初始化只影响当前 Python 配置对象，不会修改 `.env` 或操作系统环境变量，主要用于测试和嵌入式调用。

### 1.2 各配置文件的职责

| 文件 | 职责 | 是否提交 Git |
|---|---|---|
| `.env` | 本机 Secret 和环境差异 | 否 |
| `.env.example` | 环境变量模板，不含真实 Secret | 是 |
| `config/default.yaml` | 可公开的项目默认值 | 是 |
| `src/memopilot/config.py` | 配置字段、类型、加载顺序和校验规则 | 是 |

本地使用者只需要编辑 `.env`，不需要同时维护多份配置。阶段 7 将增加 `uv run memopilot setup`，参考原型 Setup Wizard 一次生成本地配置。

### 1.3 三层启动校验

1. **类型与范围校验**：例如迭代次数必须是正整数、超时必须大于 0。
2. **路径边界校验**：数据库、记忆、上传和恢复目录必须位于 workspace 内。
3. **运行完整性校验**：完整服务启动前集中检查 DeepSeek、Embedding 和飞书必填项。

配置对象允许部分字段为空，是为了让迁移、局部脚本和单元测试不必伪造所有 Secret；完整服务再调用 `validate_runtime_ready()`。

### 1.4 与原型的差异

原型以 `config.toml` 为主要入口，支持 `${ENV_VAR}` 插值和交互式 Setup Wizard，适合单进程本地应用。MemoPilot 改为类型化 Settings 和环境变量覆盖，主要服务 App、Worker、Scheduler 三进程及后续 Docker 部署。

收益是类型安全、Secret 注入和部署一致；代价是配置来源更多，因此固定“本地只改 `.env`”的使用规则，并在阶段 7 补配置向导。

## 2. 三个 SQLite 数据库

### 2.1 数据库职责

| 数据库 | 保存内容 | 一致性边界 |
|---|---|---|
| `operational.db` | Inbox、Session、Job、Run、Attempt、Step、Outbox、Fencing、Effect、消息和调度状态 | 任务执行与可靠投递事实 |
| `memory2.db` | 记忆项、Embedding、关键词索引元数据、强化和替换关系 | 长期记忆与检索事实 |
| `wake.db` | Source Event、Cursor、Reservoir、Wake Decision、Hazard、Drift、ACK | 主动唤醒事实 |

拆分依据不是表数量，而是事务边界、写入并发和数据生命周期。

### 2.2 为什么不放在一个数据库

SQLite 在 WAL 模式下允许读写大部分并行，但同一个数据库文件同一时间仍只有一个 Writer 提交。拆库后，记忆优化写 `memory2.db` 时不会占用 `operational.db` 的写锁，Scheduler 写 `wake.db` 时也不会阻塞 App 接收入站消息。

代价是不能把跨库修改当成一个简单原子事务，因此跨领域更新使用：

```text
事实库事务 + OutboxEvent
→ 异步消费
→ 稳定业务 ID 幂等写入目标库
```

### 2.3 与原型的差异

原型实际也使用多个文件，如 `sessions.db`、`memory2.db`、`proactive.db`、`consolidation_writes.db` 和 `observe.db`。MemoPilot 将分散存储整理成 operational、memory、wake 三个明确领域，便于三进程部署、迁移和面试解释。

## 3. SQLite 连接策略

### 3.1 WAL

WAL 将写入先追加到日志，Reader 可以继续读取稳定快照。它改善读写并发，但不会让同一数据库的多个 Writer 真正并行，因此事务必须短小。

### 3.2 busy timeout 与有界重试

连接设置 5 秒 `busy_timeout`，遇到短暂写锁先等待。迁移等关键路径只对 `locked/busy` 做最多三次指数退避，不重试语法错误等确定性失败。

### 3.3 外键与同步策略

- 每个连接显式执行 `PRAGMA foreign_keys = ON`。
- 使用 `journal_mode=WAL`。
- 使用 `synchronous=NORMAL` 平衡本地服务性能与可靠性。

## 4. 正确的事务范围

不能把 LLM 和工具调用包在一个长事务中。正确做法是多个短事务：

```text
事务 1：接收入站 + 创建 Job + 创建 Outbox
事务外：LLM 与工具执行
短事务：Step 与心跳
事务 2：提交 Run/Job 终态 + 后续 Outbox
```

Job 与对应 Outbox 必须在同一个数据库事务中，否则可能出现“Job 已提交但发布意图丢失”。

## 5. Schema 迁移

### 5.1 版本与原子性

数据库使用 `PRAGMA user_version` 保存 Schema 版本。每次迁移执行：

```text
BEGIN IMMEDIATE
→ 执行全部 SQL
→ 更新 user_version
→ COMMIT
```

任意语句失败则 `ROLLBACK`，数据库保持旧版本，不留下半套 Schema。

### 5.2 备份和降级保护

- 已有数据库升级前使用 SQLite Backup API 创建带旧版本和时间戳的备份。
- 不直接复制 WAL 模式下正在使用的主文件。
- 数据库版本高于当前代码支持版本时拒绝启动，不自动降级破坏数据。

## 6. Embedding 数据库身份

`memory2.db` 绑定：

```text
Embedding Base URL
+ Embedding Model
+ Vector Dimension
```

只检查维度不够：两个模型都输出 1024 维，并不代表它们位于同一个向量空间。混用后数学计算可以执行，但召回语义会静默失真。

更换模型时应重建全部向量和索引，不能只修改 `.env` 后继续使用旧数据库。

## 7. 为什么第一版使用 SQLite 向量扩展

MemoPilot 是单用户、本地优先 Agent，预期记忆量为几千到几万，而不是百万级多租户向量。阶段 4 使用 `sqlite-vec` 可以让记忆文本、向量、关系和模型元数据留在一个事实库中，减少跨库一致性和部署成本。

出现百万级向量、多用户高并发、集中式远程访问或 SQLite 写入成为实测瓶颈时，再评估 PostgreSQL + pgvector、Qdrant、Milvus 等方案。

### 阶段 0 面试表述

> MemoPilot 将任务、记忆和主动唤醒拆成三个 SQLite 事实库，通过 WAL 和短事务满足单用户三进程并发。配置使用类型化 Settings 和环境变量注入，迁移具备版本、备份、回滚和降级保护。记忆向量与 Provider、模型和维度绑定，避免配置变化造成静默检索污染。

---

# 阶段 1：可靠任务投递与会话执行权

## 1. SQLite 与 Redis 的职责

```text
SQLite = 事实源
Redis  = 临时协调层
```

SQLite 保存 Job、Run、Step、Outbox、Fencing 和副作用状态。Redis 保存优先级 Stream、Pending、Lease、队列镜像和运行时协调信息。Redis 被清空后允许从 SQLite 恢复；不能反过来把 Redis 当业务事实。

## 2. Transactional Inbox / Outbox

### 2.1 入站事务

收到消息时在一个 operational 事务中：

```text
插入唯一 inbound_event
→ 更新 session_activity
→ 创建唯一 AgentJob
→ 创建唯一 OutboxEvent
→ COMMIT
```

重复 `event_id` 或 `message_id` 返回已有结果，不创建第二个业务任务。

### 2.2 OutboxDispatcher

Dispatcher 从 SQLite 认领待发布 Outbox，再向 Redis Stream 执行 `XADD`，成功后把 Outbox 标记为 published。

如果发生：

```text
XADD 成功
→ published 尚未写回 SQLite
→ 进程崩溃
```

恢复后可能重复发布，但消费者使用稳定 `job_id` 幂等吸收，因此选择“至少一次投递”，不承诺无法证明的“恰好一次”。

### 2.3 QueueReconciler

Redis 被清空或队列镜像缺失时，Reconciler 对照 SQLite 中未完成 Job，把已有 published Outbox 重置为 pending，不创建第二条业务 Outbox。

## 3. Redis Streams 关键概念

| 命令/结构 | 作用 |
|---|---|
| `XADD` | 向 Stream 发布消息 |
| Consumer Group | 多个消费者共享消费进度 |
| `XREADGROUP` | 消费新消息 |
| Pending | 已交给消费者但尚未 ACK 的消息 |
| `XPENDING` | 查看 Pending 消息、owner 和 idle 时间 |
| `XCLAIM` | 把指定 Pending 消息交给另一个消费者实例 |
| `XACK` | 告诉 Group 该消息已完成 |
| `XDEL` | 从 Stream 删除终态消息 |

`XACK` 只清除 Consumer Group 的 Pending 状态，不等同于业务成功；业务终态必须先提交 SQLite。

## 4. PendingMessageReclaimer

Worker 取走消息后崩溃，重启后的 Worker 实例只有同时满足以下条件才允许接管：

```text
Pending idle 超过阈值
+ Redis Lease 已消失
+ SQLite Run 心跳已过期
```

然后使用 `XPENDING + XCLAIM` 精确认领。第一版不直接使用 `XAUTOCLAIM`，因为后者扫描时可能先改变 owner，再检查会话 Lease 和 SQLite 心跳。

Worker A 和 Worker B 通常表示前后启动的两个进程实例，不代表系统长期运行多个 Worker 副本。第一版默认一个 Worker，新的实例可能来自崩溃重启、重新部署或误启动重叠。

## 5. Lease

Lease 是 Redis 中带 TTL 的会话执行权：

```text
session_key + owner_id + epoch
```

用途是让同一会话同一时间只有一个执行者，并阻止用户正在聊天时主动任务同时进入 Agent Runtime。

Lease 可能因为进程崩溃、Redis 断连、事件循环阻塞、机器休眠或续租失败而过期。短暂续租失败不等于立即失权；Worker 会在安全窗口内重试，但不能在无法确认所有权时继续发起新副作用。

## 6. Fencing

Redis Lease 只能表示“当前看起来谁持有锁”，无法保证旧进程一定死亡。Fencing Epoch 由 SQLite 单调递增：

```text
Worker A：epoch=5
Worker B 接管：epoch=6
```

所有关键写入校验 owner 和 epoch。Worker A 即使恢复，也无法用 epoch=5 覆盖 epoch=6 的状态。

核心区别：

```text
Lease   = 减少并发执行
Fencing = 拒绝失权者写入
```

## 7. Job、Run、Attempt、Step 与执行身份

| 字段 | 含义 |
|---|---|
| `session_key` | 哪个会话 |
| `job_id` | 哪项业务任务 |
| `run_id` | 该任务的运行与审计记录 |
| `attempt_no` | 第几次进程执行尝试 |
| `owner_id` | 哪个 Worker 实例 |
| `fencing_epoch` | 第几代执行权 |

可以记成：

```text
session / job / run = 正在处理什么
owner / epoch        = 现在轮到谁处理
```

## 8. ACK 与终态清理

Worker 必须先把 Job/Run 终态提交 SQLite，再执行 Redis 清理：

```text
SQLite succeeded/failed/needs_review
→ XACK
→ XDEL
→ 从 queued:job_ids 镜像移除
```

如果在 SQLite 终态提交后、ACK 前崩溃，恢复器只执行 cleanup，不再次运行 Agent。

## 9. 外部副作用状态

| 状态 | 含义 | 是否可自动重试 |
|---|---|---|
| `confirmed` | 已知成功 | 不需要 |
| `failed` | 已知失败 | 按策略决定 |
| `unknown` | 请求已发出但结果不明确 | 默认不能 |
| `needs_review` | 系统无法自动消除不确定性 | 人工或专门审核流程 |

如果服务支持稳定幂等键，继任者使用相同业务键和相同请求内容重放，才可以避免重复副作用。非幂等操作在“服务可能已执行、本地未记录”时不能标记为普通 failed。

### 阶段 1 面试表述

> MemoPilot 采用 SQLite 事实源加 Redis 协调层。Transactional Outbox 保证业务提交后可恢复发布，Redis Streams 提供至少一次投递，Lease 负责会话串行，SQLite Fencing 拒绝僵尸 Worker 写入。恢复器通过 Pending idle、Lease 和 Run 心跳三重条件进行保守接管，外部副作用结果不明确时进入 needs_review 而非盲目重放。

---

# 阶段 2：可追踪 Agent Runtime

## 1. 外层 Phase 与内层 ReAct

完整 Turn：

```text
BeforeTurn
→ BeforeReasoning
→ PromptRender
→ ReAct 循环
   ├─ BeforeStep
   ├─ LLM
   ├─ Tool Calls
   └─ AfterStep
→ AfterReasoning
→ AfterTurn
```

前三个 Phase 每个 Turn 一次；BeforeStep/AfterStep 随每次模型调用重复；最后两个 Phase 在 ReAct 结束后各执行一次。

## 2. 七个 Phase 的职责

| Phase | 当前职责 | 后续扩展位置 |
|---|---|---|
| BeforeTurn | 校验会话和输入 | 中断状态、Turn 初始化 |
| BeforeReasoning | 准备推理输入 | 自动记忆预检索、偏好、Skills |
| PromptRender | 组合 system、history、当前用户消息 | PromptBlock 注入 |
| BeforeStep | 每轮模型调用前准备并审计 | Token、工具可见性、Lease 再校验 |
| AfterStep | 记录模型与工具 Observation | 循环检测、进度、插件 Hook |
| AfterReasoning | 整理 ReActResult | 保存消息、触发异步归档 |
| AfterTurn | 完成业务 Turn | Outbox、清理和后续任务 |

## 3. Phase DAG 合同

每个模块声明：

```text
slot     = 模块唯一标识
requires = 执行前必须存在的数据
produces = 执行后承诺生成的数据
```

Runtime 按依赖拓扑排序，而不是依赖注册偶然顺序。MemoPilot 在启动时拒绝缺失依赖、循环、重复 slot 和产物冲突；`provided_slots` 还声明动态数据从哪个生命周期才开始可用，防止 BeforeTurn 错误依赖未来的 `step.response`。

原型对部分缺少依赖的可选模块会选择禁用；MemoPilot 当前对核心 Runtime 合同更严格。后续插件阶段需要明确区分核心模块与可选模块。

## 4. ReAct 与 Function Calling

一轮 ReAct：

```text
LLM 提出 Action（Tool Call）
→ ToolRegistry 执行
→ 产生 Observation
→ Observation 回填给 LLM
→ LLM 决定重试、换工具或自然语言结束
```

工具不存在、参数无效和工具异常都转换成失败 Observation，不直接终止 ReAct。Lease/Fencing 失权和 Provider 不可用属于基础设施错误，不能交给 LLM 决定执行权。

## 5. ToolRegistry

ToolRegistry 负责：

- 工具注册与重名检查。
- 向模型暴露 JSON Schema。
- 校验模型参数，不擅自进行语义类型转换。
- 执行异步 Python 工具。
- 把未知工具、参数错误和执行异常转成结构化 Observation。

模型只生成 Function Call，真正执行 Python 函数的是 Runtime。

## 6. 达到迭代上限后的自然语言收尾

达到最大工具轮次后，Runtime 额外调用一次 LLM，并传入空工具列表：

```text
已有消息和工具结果
+ 收尾提示
+ tools=[]
→ 自然语言说明已完成、未完成和当前结论
```

清空工具列表的主要原因不是节省 Token，而是强制终止新的 Tool Call 和外部副作用。

## 7. Provider 适配边界

```text
DeepSeek/OpenAI-compatible SDK
↕ OpenAICompatibleProvider
统一 ModelResponse / ChatMessage
↕ ReAct
```

更换模型厂商时主要修改 Provider，不让 ReAct 出现 `if provider == deepseek`。

### DeepSeek reasoning_content

思考模式产生 Tool Call 时，DeepSeek 要求后续请求回传 `reasoning_content`。MemoPilot 按原型实现：

```text
Provider 读取 reasoning_content
→ 放入不透明 provider_fields
→ ReAct 随 assistant Tool Call 原样携带
→ DeepSeek Provider 序列化时重新发送
```

思考模式默认关闭，可通过配置开启。开启时 Provider 修补历史 assistant 消息；非 DeepSeek Provider 默认剥离专属字段。SQLite 运行审计不保存思考正文。

## 8. RuntimeJobExecutor

AgentRuntime 只负责一次对话推理，不直接操作 Job、Redis 或 SQLite 终态。RuntimeJobExecutor 负责：

```text
校验 RunClaim / Lease / Turn 身份
→ 创建绑定当前 Run 的 StepSink
→ 调用 AgentRuntime
→ 把 Runtime 结果映射为 Job 成败
→ 使用 Fencing 提交终态
```

本地 CLI 不经过可靠任务系统时可以直接调用 AgentRuntime。

## 9. Provider 失败与 Job 失败

Provider 超时后会返回固定自然语言兜底，避免用户看到 Python 异常，但 Job 仍标记 failed：业务目标是回答用户问题，而不是“成功生成任意字符串”。

真正修改 SQLite Job 状态的是 RuntimeJobExecutor 调用 OperationalRepository，不是 LLM、Provider 或 Redis。

## 10. ReAct 迭代与 SQLite Step 的区别

- ReAct Step：一轮模型决策。
- SQLite `steps`：一条 Phase、模型或工具审计事件。

一轮 ReAct 可能对应多条 SQLite Step，因此崩溃后可以看到最后完成的细粒度事件。

## 11. 审计日志不等于安全 Checkpoint

当前阶段 2 已实现 Step 审计，但尚未完整实现跨进程精确断点恢复。安全 Checkpoint 还需要：

- 完整 ChatMessage 消息链。
- Tool Calls、Call ID 和 provider_fields。
- 每个工具 Observation。
- 迭代编号和完整性标记。
- 副作用的 confirmed/unknown 状态。
- 恢复时的 Fencing 身份。

只有最后一个完整 AfterStep Checkpoint 且所有副作用都可证明安全时，继任 Worker 才能从下一轮 BeforeStep 继续。结果不明确的非幂等副作用进入 needs_review。

## 12. 原型 `/stop` 续跑与精确 Checkpoint

原型 `/stop` 保存纯内存 `TurnInterruptState`，把原任务、部分回复、已使用工具和用户补充拼成新的输入，再启动一个新 Turn。这是语义续跑，不是进程崩溃后从某个 Tool Call 精确恢复。

MemoPilot 计划在阶段 3 保留该体验。由于 App 和 Worker 分进程，中断信号和带 TTL 的临时快照需要通过 Redis 协调，行为仍保持原型语义。

### 阶段 2 面试表述

> MemoPilot 迁移原型七阶段生命周期和 ReAct Function Calling 合同，但保持编排代码可读，不依赖 LangGraph 黑盒。Runtime 对 Phase 依赖进行启动诊断，工具失败作为 Observation 回到模型，每轮模型和工具事件使用当前 Fencing 身份写入 SQLite。Provider 隔离模型厂商差异，RuntimeJobExecutor 隔离可靠任务基础设施。

---

# 阶段 3：飞书私聊、实时反馈与可靠发送

## 1. 原型实际做法与 MemoPilot 迁移边界

原型飞书使用 SDK 长连接接收事件，处理文本、富文本、图片、身份映射和 `/stop`；当前飞书 `send_stream()` 主要仍委托普通发送。原型中更成熟的思考过程、工具状态和临时消息更新来自 Telegram 实时反馈链路。

MemoPilot 保留飞书私聊长连接与 `/stop` 的核心语义，同时把原型 Telegram 的实时反馈语义迁移为通用 `ReActProgressObserver`，再由 `FeishuLiveProgress` 渲染飞书卡片。因此这部分不是逐行复制原型飞书实现，而是“原型交互语义 + 飞书卡片 + 三进程可靠性边界”的适配。

## 2. 入站与中断链路

```text
飞书长连接事件
→ 进程内事件去重
→ 白名单与 p2p 私聊过滤
→ 文本/图片解析和身份映射
→ 普通消息：InboundBridge → SQLite Inbox/Job/Outbox
→ /stop：OperationalInterruptController → SQLite 中断请求 → 立即确认
```

进程内去重减少同一 App 生命周期中的重复解析；SQLite Inbox 才是跨重启的持久化幂等事实源。`/stop` 不能作为普通 P0 Job 排队，否则会等待当前 Job 结束后才执行，失去中断意义。App 将中断请求持久化并通过 Redis 快速通知 Worker；Worker 同时轮询 SQLite 作为信号丢失后的兜底，取消 Runtime 后保存原问题、部分回复、已完成工具和 Observation 组成的 `TurnInterruptSnapshot`。下一条普通消息通过 `reserve → consume/release` 两阶段领取快照，创建新 Turn 做语义续接，而不是恢复旧 Python 调用栈。

即时回执只表示中断请求已经可靠受理，因此文案为“已收到停止请求，正在中断本轮任务”，不能在 Worker 完成取消前宣称“已中断”。

## 3. 流式反馈与两种 Observer

`StreamDelta` 区分 `thinking_delta` 与 `content_delta`。Provider 一边发送增量，一边累计完整 `ModelResponse`；Function Calling 的名称和参数必须等流结束、JSON 拼接完整后才能交给 ReAct 执行。

- `ReActObserver`：`before_step/after_step`，承担审计、中断和 Checkpoint 等正确性职责，异常不能默认忽略。
- `ReActProgressObserver`：模型增量、工具开始、工具完成，只承担界面反馈，异常由 `_BestEffortProgress` 隔离。

`FeishuLiveProgress` 是当前具体实现，但 Runtime 不依赖飞书；以后可替换为 Telegram、CLI 或 WebSocket 进度端。

## 4. 实时卡片的可靠性边界

首个进度事件创建卡片，后续通过 `message_id` 修改同一张卡片。实现包含时间/字符节流、异步锁、429 退避、失败预算和自动降级。达到失败或限流预算后只关闭本轮实时预览，ReAct 与最终文本链路继续运行。

每次创建或更新前检查 Run/Job 状态、Lease、Fencing Epoch 和授权策略；失权 Worker 不能继续污染界面。`finalize()` 只定格过程摘要，最终自然语言答案通过独立可靠发送链路完成。

## 5. Outbound Effect 与最终 Job 状态

```text
ReAct 最终答案
→ SQLite 创建稳定 operation_id 的 Effect
→ 校验发送资格
→ 飞书发送
→ confirmed / failed / unknown / cancelled
→ succeeded / failed / needs_review / cancelled
```

网络超时无法证明飞书未收到消息，因此 Effect 进入 `unknown`、Job 进入 `needs_review`，不能盲目自动重发。确认远端未发送后才能使用相同 `provider_uuid` 显式重试；确认远端已经发送时，应把 Effect 和 Job 对账为 `confirmed/succeeded`，不得再产生发送副作用。

## 6. 学习中发现并完成的调优

1. **按 Job 来源授权**：普通消息只排队，不因 `activity_version` 增长废弃旧被动回复；主动推送发现用户产生新活动时取消本轮，等待下次唤醒；`/stop` 才能抢占被动 Job。
2. **Effect 人工处置闭环**：增加 `list/show/confirm/fail/retry` 本地运维入口。`confirm` 必须提供真实飞书 `message_id`；`retry` 只用于已确认远端未发送的场景，并复用稳定 `provider_uuid`。
3. **App/Worker 启动装配**：阶段 3 已提供两个常驻入口；Scheduler 仍属于后续主动任务阶段。Effect 命令是一次性运维工具，不是第四个进程。
4. **跨进程中断**：补齐 Worker 消费、Runtime 取消、Snapshot 持久化和下一条消息语义续接，并修正即时回执的 accepted/completed 语义。
5. **异步工具合同**：Redis 中断信号无法解决 asyncio 事件循环被同步工具阻塞的问题。ToolRegistry 只接受真正异步的 Handler，并提供统一超时；Shell 等阻塞或强副作用能力必须使用可终止异步子进程，不能用取消后仍可能存活的线程包装。

## 7. 已验证与不可宣称范围

本地单元/集成测试已覆盖事件解析、私聊过滤、去重、`/stop`、Snapshot 续接、卡片 JSON、创建/更新/限流降级、Provider 流式增量、工具生命周期、最终 Effect 状态机和人工对账入口。在用户明确许可下完成过真实飞书长连接验收，手机端确认了思考卡片、最终文本、A/B 排队和 `/stop` 中断续接行为。

后续未经用户针对当次测试明确许可，不得再次调用真实飞书、DeepSeek、Embedding 或其他外部 API。当前不能把一次人工验收扩大表述为已经覆盖真实限流、网络分区和所有 SDK 重连故障。

### 阶段 3 面试表述

> MemoPilot 使用飞书长连接承接私聊入站，把 SDK 回调线程安全桥接到 App 的 asyncio 主循环，消息持久化后再结束回调。模型增量和工具生命周期通过通用 Progress Observer 渲染为可降级实时卡片；最终回复走独立的 SQLite Outbound Effect，并用稳定业务键、Lease 和 Fencing 管理外部副作用。普通消息按队列顺序回答，主动推送根据 activity_version 为用户让路，`/stop` 通过 Redis 快速通知与 SQLite 兜底完成跨进程取消和语义续接；发送结果不明确时进入 needs_review，由本地命令人工对账而不盲目重放。

---

# 阶段 4：分层记忆、异步归档与可恢复优化

## 1. 三层记忆不是三份重复数据

- 近期上下文由 SQLite 原始消息和 `RECENT_CONTEXT.md` 承担，解决当前对话连续性。
- Markdown 档案使用 `MEMORY.md`、`SELF.md`、`HISTORY.md`、`RECENT_CONTEXT.md`、`PENDING.md` 和每日日志保存人类可读的长期结论、历史、自我认知、近期背景与候选事实。
- `memory2.db` 保存带类型、来源、时间、Embedding 和状态的原子记忆，解决跨措辞的语义召回。

原始消息保存证据，Markdown 保存整理后的叙事，向量层保存可检索的事实；同一事实可能在生命周期中出现于多个位置，但职责不同。

原型实际也是近期上下文、Markdown 档案和结构化向量记忆并存。MemoPilot 没有改变这套记忆语义，主要增加了三进程环境所需的 SQLite Job、Redis 派发、Manifest、Lease 和 Fencing。

## 2. 预检索与 `recall_memory`

预检索在 ReAct 前自动执行，默认使用原始 Query，不启用 HyDE，用较低成本提供基础上下文。`recall_memory` 由 ReAct 在信息不足时主动调用，支持 `answer`、`timeline`、`interest` 和 `procedure` 意图；其中 `answer` 保留原型的事件型、一般事实型双假设增强，`timeline` 必须提供明确时间范围。

## 3. 异步 Consolidation 与 Vectorization

成功 Turn 提交后异步创建 `memory.consolidate`，避免把额外 LLM 延迟加入用户回复。当前链路明确拆成两步：

1. `memory.consolidate` 负责基础归档。达到归档阈值时，Consolidation LLM 只生成 `history_entries` 和 `pending_items`，Recent Context LLM 使用独立 Prompt 更新 `RECENT_CONTEXT.md`；随后统一提交 HISTORY、PENDING、RECENT_CONTEXT、每日日志和 Manifest。
2. `memory.vectorize` 从已提交 Manifest 中恢复原始对话。`event` 直接来自稳定的 `history_entries`；第二个长期记忆 LLM 只提取 `profile/preference/procedure`，结果先持久化到 Manifest 的 `_implicit_memories`，再逐条写入向量库。

“Recent Context 独立调用”是指它使用独立的一次 LLM 请求，不是脱离 `memory.consolidate` 的另一个 Redis Job。没有达到归档阈值时，两次 LLM 都不会调用；代码只从 SQLite 读取最近消息并替换 `RECENT_CONTEXT.md` 的 `## Recent Turns` 区块，用户消息保留全文，Assistant 只保留短预览，不消耗 Token。

Vectorization 根据稳定来源和内容哈希幂等写入：同一来源重复执行返回 `unchanged`，同一事实来自新来源时强化原条目而不是复制。SQLite 与 `sqlite-vec` 满足当前单用户规模，同时保留余弦回退和索引重建能力。

拆分后的失败边界是：基础 Markdown 归档成功后，即使长期记忆提取或向量写入失败，HISTORY、PENDING、RECENT_CONTEXT 和原始消息仍然存在；Worker 只重试 `memory.vectorize`。隐式记忆提取结果已先写回 Manifest，因而写向量途中崩溃也不需要再次调用 LLM。

## 4. MemoryOptimizer

Optimizer 将当前 `PENDING.md` 原子移动为快照，优化期间的新候选继续进入新的 `PENDING.md`。模型分别生成完整的 `MEMORY.md` 和 `SELF.md`，发布清单把 `MEMORY.md`、`SELF.md` 和 `HISTORY.md` 三个文件纳入同一个可恢复提交边界：

- 模型失败或发布状态仍为 `writing`：恢复旧版 `MEMORY.md`、`SELF.md`、`HISTORY.md`，把快照放回 Pending，等待整批重试。
- 发布状态为 `committed`：保留新版 MEMORY/SELF，把本批 Pending 快照归档进 HISTORY，再删除已消费快照。
- Worker 丢失 Lease：禁止提交旧计算结果，留给持权 Worker 恢复。

优化期间新写入的 Pending 不属于当前快照，不会被本轮误归档。`MemoryMaintenanceScheduler` 已在 Scheduler 装配入口注册，能够周期创建幂等的 `memory.optimize` Job；可以宣称具备自动周期优化，但不能把离线测试扩大成长期生产稳定性证明。

## 5. 混合检索、热度与逻辑过期

向量通道同时保留原始语义分数 `semantic_score` 和加入 hotness 后的综合 `score`，关键词通道补足项目名、命令和 ID 等精确命中，RRF 只根据两条通道中的相对排名融合结果并另写 `rrf_score`。

阶段复盘曾发现 `_rrf_merge()` 用约 `0.02` 的 `rrf_score` 覆盖原始相关性，导致普通记忆召回成功却无法通过注入阈值。现在已用真实 `retrieve -> RRF -> injection` 组合测试修复：`semantic_score` 用于绝对语义阈值，`rrf_score` 只负责多路候选融合排名，hotness 参与通过阈值后的排序；不再使用“必须接近本轮最高分”的自创相对分数门槛。

hotness 对齐原型：强化次数经过 `log1p + sigmoid` 得到频率项，访问时间按指数衰减得到新近项，情绪强度会延长有效半衰期，最后用可配置的 `alpha` 与语义分数加权。系统先按原始语义分数过滤低相关结果，再使用 hotness 重排，避免“经常访问但不相关”的记忆越过相关性门槛。

记忆过期采用逻辑失效而非时间硬删除。显式纠错通过 `forget_memory` 将稳定 ID 标记为 `superseded`；后台向量化写入 Preference、Procedure 和状态/购买类 Profile 时，会以原始语义相似度识别旧项，并在同一事务内写入新项、退役旧项。普通替换阈值为 `0.90`；对于 `emotional_weight >= 7` 的旧 Profile 状态/购买记忆，原型和 MemoPilot 都提高到 `0.92`，避免重要旧状态被仅仅近似的新描述过早覆盖。再次确认完全相同的事实会重新激活原稳定 ID。这样保留审计链和历史依据，同时默认检索只返回 active 记忆。

注入阶段把候选分为强制程序规则、偏好/流程和相关历史，并分别限制条数和总字符。普通记忆低于绝对语义阈值不注入；刚超过阈值但未达到 `threshold + 0.15` 时标记“有印象，不确定”，提醒主 Agent 不要把边缘匹配当成确定事实。带 `tool_requirement` 的 Procedure 一旦进入候选集合便作为强制约束注入，不再被普通语义阈值挡掉。注入内容还携带记忆 ID、发生时间、距今时间和证据来源，便于引用审计。

## 6. PostResponse：回复后的旧记忆纠错

最终回复完成并提交后，同一事务创建 `memory.post_response` Job，由 Memory Worker 异步执行，不阻塞飞书回复。它从 SQLite 读取本轮用户原话，提取最多 9 个潜在纠错主题，只检索相关的 Preference 和 Procedure，再让低预算 LLM 判断哪些旧记忆已被用户明确推翻；命中的旧项标记为 `superseded`。

PostResponse 不创建新记忆，不修改 Event/Profile，也不会因为自身失败把已成功回复的 Job 改成失败。新事实由本轮显式 `memorize` 或后续 Vectorization 写入。系统还会从本轮 Step 审计中恢复 `memorize` 成功返回的记忆 ID，把它们加入保护集合，避免“刚写入就被同轮 PostResponse 退役”。

## 7. 你重点追问过的问题

> [!IMPORTANT]
> **Consolidation LLM 和 Recent Context LLM 都只有达到归档阈值才调用吗？**
>
> 是。未达到阈值时不调用任何归档 LLM，只用普通代码刷新 `## Recent Turns`。达到阈值时，两次调用都发生在同一个 `memory.consolidate` Job 内，但使用不同 Prompt、承担不同职责。

> [!IMPORTANT]
> **“只用代码更新 Recent Turns”是什么意思？**
>
> 从 SQLite 查询最近几条原始消息，按固定 Markdown 模板替换 `RECENT_CONTEXT.md` 的指定区块；不理解、不总结、不改写，也不消耗模型 Token。

> [!IMPORTANT]
> **为什么 Consolidation 不再直接提取四类向量记忆？**
>
> 基础归档必须优先稳定落盘。把隐式长期记忆提取放到可重试的 Vectorization 后，模型失败不会阻塞 HISTORY/PENDING/RECENT_CONTEXT；提取结果先存 Manifest，重试也不会反复花 Token。

> [!IMPORTANT]
> **RRF、语义阈值和 hotness 分别负责什么？**
>
> `semantic_score` 先判断“是否相关”，RRF 融合向量与关键词通道的相对排名，hotness 在相关结果中体现强化次数、新近程度和情绪保护。RRF 不设语义阈值，热度也不能把不相关记忆抬过相关性门槛。

> [!IMPORTANT]
> **预检索和 `recall_memory` 有什么区别？**
>
> 预检索在 ReAct 前自动提供一批基础记忆；`recall_memory` 是 ReAct 判断信息不足后主动调用的工具，可以显式选择 `answer/timeline/interest/procedure` 意图并补充时间范围。前者降低每轮主动检索成本，后者负责按任务深入追忆。

> [!IMPORTANT]
> **“有印象，不确定”有什么用？**
>
> 它是写给主 Agent 的软风险提示：允许参考刚过阈值的边缘记忆，但不能据此武断陈述或执行高风险动作。它不修改数据库状态，也不是代码级安全保证。

> [!IMPORTANT]
> **为什么强制 Procedure 可以绕过普通注入阈值？**
>
> 带 `tool_requirement` 的 Procedure 表示用户明确要求必须采用某工具或流程。它进入候选集合后属于行为约束，不应再按普通背景记忆处理；普通 Procedure 和 Preference 仍受语义阈值限制。

> [!IMPORTANT]
> **记忆没有真正注入 Prompt，为什么不能仍然算作引用？**
>
> 因为“检索到候选”不等于“主 Agent 看见并使用”。现在只有最终注入块中仍然可见的稳定记忆 ID 才进入引用审计；被阈值、条数或字符预算过滤掉的候选不能强化，也不能宣称被回答引用。

> [!IMPORTANT]
> **PostResponse 到底怎么服务？**
>
> 它在回复完成后异步清理被本轮用户明确推翻的旧 Preference/Procedure，不负责回复，也不负责创建新记忆；同轮显式 `memorize` 的新记忆会被保护。

> [!IMPORTANT]
> **每日日志会不会从多个来源重复覆盖？**
>
> 不会。日记只从 Consolidation 规范化后的 `history_entries` 按 `[YYYY-MM-DD HH:MM]` 分组生成，不再同时解析 HISTORY 标题和结构化 Event，避免同一天文件被两条链路重复写入。

> [!IMPORTANT]
> **Optimizer 期间崩溃或又有新 Pending 怎么办？**
>
> 当前批次先形成独立快照，新 Pending 写入新文件。崩溃时按发布清单恢复 MEMORY/SELF/HISTORY 并放回快照；成功时只归档当前快照，新写入内容留给下一批。

> [!IMPORTANT]
> **记忆合并和失效适用于所有类型吗？**
>
> 不是完全相同：Event 更重视独立事件与时间窗口；Preference、Procedure 和部分状态型 Profile 支持相似新事实替换旧事实；高情绪旧状态提高替换门槛；默认检索只返回 `active`，旧记录保留用于审计而不是物理删除。

## 8. 已验证与不可宣称范围

阶段 4 完整离线验收为 285 个测试通过，Ruff、MyPy 和差异检查通过。覆盖三层记忆、预检索与 `recall_memory`、Consolidation/Vectorization、Manifest 断点、显式记忆、PostResponse、引用审计、混合检索、热度与逻辑失效、Optimizer 发布恢复和周期调度。

未经用户针对当次任务明确授权，不能调用真实 DeepSeek、Embedding、飞书或其他外部 API。当前不能宣称已验证百万级多租户向量规模、长期生产运行、真实网络分区下的全部恢复路径，也不能把软提示“有印象，不确定”描述成强安全保证。

### 阶段 4 面试表述

> MemoPilot 保留原型的近期上下文、Markdown 档案和结构化向量记忆三层模型，将基础归档、隐式长期记忆提取、向量写入、回复后纠错和周期优化拆成可独立失败、可幂等重试的后台任务。SQLite 保存消息、任务、Manifest 和记忆事实，Redis 只负责派发；稳定业务 ID、Pending 快照、发布清单以及 Lease/Fencing 共同保证重复投递和进程崩溃不会静默丢失或重复记忆。混合检索先用绝对语义阈值保证相关性，再通过 RRF 融合多路召回、用 hotness 重排，并将证据、时间和低置信度提示注入主 Agent。

---

# 阶段 5：插件、Skills、MCP 与工具筛选

## 1. 三种扩展机制的职责边界

插件、Skill 和 MCP 都能扩展 Agent，但扩展的层次不同：

- **Plugin 改系统**：可信 Python 代码运行在 Worker 进程内，可以注册工具、拦截工具调用、订阅生命周期事件或插入七阶段模块，从而改变 Agent Runtime 的行为。
- **Skill 教模型**：以 `SKILL.md` 保存可复用的任务说明、工具使用约束和操作流程，通过 Prompt 注入影响模型决策，本身不执行代码。
- **MCP 给工具**：独立 MCP Server 通过标准协议向 Agent 暴露外部工具；MemoPilot 第一版只实现本地 stdio Client，不允许 MCP 直接插入内部生命周期。

面试时可以概括为：Plugin 扩展运行时，Skill 提供领域工作流，MCP 标准化外部能力接入。插件也可以注册工具，但它与 MemoPilot 的 Python 接口紧耦合；MCP Server 可用其他语言独立实现。

## 2. 原型式插件合同与四个扩展点

MemoPilot 以原型的 `Plugin` 子类、装饰器、`PluginContext` 和 `PluginManager` 为主体迁移，没有继续维护无法形成 Python 沙箱的强制 capability manifest。当前四个主要扩展点是：

1. `PhaseModule`：插入七阶段模块链，使用 `slot/requires/produces` 表达执行顺序和数据依赖，适合主流程节点。
2. EventBus 装饰器：把函数注册为类型化生命周期 Handler，适合较轻量的观察或上下文改写。
3. `@on_tool_pre`：在一次工具调用前按顺序检查或改写参数，也可以拒绝调用；前一个 Hook 的输出是后一个 Hook 和真实工具的输入。
4. `@tool`：把插件方法包装成统一 Tool，登记名称、描述、JSON Schema、`risk`、`always_on` 和搜索提示。

EventBus 和 PhaseModule 都能影响阶段行为，但抽象层不同：EventBus 按事件类型调用 Handler，不理解 Phase Frame 的 slot；PhaseModule 是 DAG 中的正式节点，适合数据依赖明确的主链路。工具 Hook 不走 EventBus，它位于统一 ToolRegistry 的工具执行路径。

## 3. 插件从定义到执行

插件函数需要区分三个时刻：

```text
模块导入：装饰器只记录元数据
→ 启动装配：PluginManager 把 Handler、Tool、Hook 和 Module 注册到对应容器
→ 运行期间：事件发生或模型调用工具时才执行函数
```

当前启动顺序为：扫描 `plugin.py`、检查 `plugin.disabled`、导入模块、创建实例、应用可选 Manifest、注入 Context、登记 Event Handler 和 Tool、收集 Hook/Phase/Prompt，最后 `await initialize()`。Event Handler 比 Tool 先登记只是原型实现顺序，不表示 Handler 已经执行或拥有更高运行优先级。

可选 `manifest.yaml` 只提供 `name/version/desc/author` 等身份信息，不是权限清单。`PluginContext` 是依赖注入容器，向插件提供 `event_bus`、`tool_registry`、`plugin_id`、插件目录、工作区、配置、KV、会话管理器和记忆引擎等运行资源。

初始化失败时，插件不进入 loaded 列表，已注册 Tool、已收集 ToolHook、PhaseModule 和 PromptBlock 会撤回并留下 diagnostic；原型式直接注册到 EventBus 的 Handler 和动态模块导入不保证完整回滚。第一版依赖“所有者信任的本地插件、启动期静态装配、不做热重载”的部署假设，不把它宣称为原子插件沙箱。

关闭时按加载顺序反向执行 `terminate()`，再注销工具并清理 Manager 持有的扩展集合；单个 `terminate()` 异常只记录 diagnostic，不阻断其他插件关闭。

## 4. 插件 Config 与 KV

插件配置由作者默认值和用户覆盖组成：

```text
_conf_schema.json：插件作者声明默认值
plugin_config.json：用户覆盖
PluginContext.config：合并后的只读使用视图
```

当前 `_conf_schema.json` 主要用于提取默认值，不应宣称具备完整 JSON Schema 类型校验。

KV 保存插件运行时产生的游标、计数和状态。原型将 `.kv.json` 放在插件源码目录；MemoPilot 改为 `workspace/.memopilot/plugin_state/{plugin_id}.json`，避免运行状态污染 Git、只读插件目录或安装 wheel。配置回答“用户希望怎样运行”，KV 回答“插件运行到了哪里”。

## 5. EventBus 的 Gate、Observer、fanout 与 enqueue

- Gate Handler 参与主流程控制或上下文改写，调用方等待结果，异常可以影响本轮。
- Observer 只观察已经发生的事件；单个 Observer 异常记录 diagnostics，不应让整个 Agent Turn 失败。
- `fanout()` 当前等待本次所有 Observer 完成，因此慢 Observer 仍可能增加调用点延迟。
- `enqueue()` 把事件放入后台队列，调用方不等待 Observer 完成；Worker 关闭时需要排空或明确停止队列。

“Observer 异常不传播”不等于“Observer 一定不阻塞”，是否阻塞取决于调用处使用 `fanout` 还是 `enqueue`。

## 6. 原型式工具筛选

大量插件和 MCP 工具不应每轮把全部 Schema 塞给模型。MemoPilot 对齐原型的三层可见集合：

```text
always_on 常驻工具
+ 当前会话 LRU 预热工具
+ 本轮 tool_search 新解锁工具
= 当前 ReAct Step 可见 Schema
```

`tool_search` 搜索 ToolDocument 中的名称、描述、参数、来源、风险和 `search_hint`，匹配结果加入当前会话的解锁状态。隐藏工具即使被模型凭名称猜中，也必须经过统一 preflight 和 Hook 审计，然后拒绝执行真实 Handler。插件动态注册与 MCP 增删会同步更新搜索索引；已注销工具即使残留在会话 LRU 名称中，也无法再取得 Schema 或 Handler。

MemoPilot 保留原型顺序和 Phase 可见性，但用 `ContextVar` 隔离并发 Turn 的瞬时搜索排除集，避免共享 Runtime 中 `already_loaded` 串线。工具筛选默认开启，可以显式关闭回到全量 Schema。

`risk` 是工具搜索、展示和风险提示元数据，不是持久化 Effect 状态机。正式筛选值为 `read-only`、`write` 和 `external-side-effect`；原型插件装饰器的默认 `read-write` 作为兼容值保留，但不属于 tool_search 的正式过滤枚举。

## 7. Skills 的装载与 Prompt 注入

Worker 在核心工具、MCP 管理工具和插件工具登记后运行 SkillLoader：

```text
src/memopilot/builtin_skills
+ workspace/skills
→ workspace 同名覆盖 builtin
→ 检查 required_tools
→ SkillCatalog
→ AgentRuntime PromptRender
```

`always: true` 写在 `SKILL.md` 的 YAML frontmatter 中，不写入给模型看的 Catalog。Catalog 展示名称、描述、来源和可用状态；Runtime 另外读取 `always`，将可用常驻 Skill 的完整正文每轮注入。用户通过 `$skill-name` 显式选择的可用 Skill也会注入全文。

MCP 或插件工具动态变化时，SkillCatalog 会刷新 `required_tools` 可用状态；它不会重新扫描磁盘，因此运行期间新建 `SKILL.md` 仍需重启 Worker。Skill 可用或激活也不会绕过 Tool Search：所需工具 Schema 尚未可见时，模型仍需先搜索并解锁工具。

原型普通 Skill 与 Drift Skill 使用相同的 `SKILL.md` 包装思想，但目录、Loader、状态和执行链不同。阶段 5 只提供普通 Skill 与 `background_allowed` 候选筛选；真正消费后台候选的 Drift/Wake 执行属于阶段 6，不是阶段 5 遗漏。

## 8. MCP stdio 主链路

MCP 是 Model Context Protocol（模型上下文协议）。MemoPilot Worker 是 Host，`McpServerClient` 是 Client，独立子进程是 Server，第一版通过 stdin/stdout 传输 MCP JSON 消息。

连接过程为：启动子进程、建立官方 SDK `ClientSession`、`initialize` 握手、发送 `initialized` 通知、调用 `tools/list`，再把远端工具包装成本地 Tool。工具别名使用 `mcp_{server_id}__{remote_name}`，防止多个 Server 或内置工具重名。

一次调用沿以下路径执行：

```text
LLM Function Call
→ ToolRegistry Schema 校验与前置 Hook
→ MCP Tool Wrapper
→ ClientSession tools/call
→ content / structuredContent / isError
→ ToolObservation
→ ReAct 决定重试、换工具、询问用户或结束
```

MemoPilot 使用官方 MCP SDK，保留 `isError`、内容块和结构化结果；协议错误、超时和 Server 不可用会映射为明确的 ToolObservation，而不是把错误伪装成成功字符串。工具失败仍遵循阶段 2 合同：先交回 ReAct，不直接使整个 Runtime 崩溃。

## 9. MCP Actor 与动态管理

每个 MCP Server 由一个专属 Actor Task 持有 SDK Session。并发调用只向其 `asyncio.Queue` 投递 Request，并通过 Future 等待对应结果；同一 Server 串行，不同 Server 可以并行。这避免原型手写 Client 中多个协程同时读取同一 stdout、误消费其他 request id 响应的竞态。

调用方取消只取消自己的等待，不关闭共享 MCP Session。活动请求发生连接错误时返回 ToolObservation，Actor 可以重建连接后继续处理队列中的后续请求；`max_restarts` 控制连接重建次数，不会自动重放已经失败的工具调用。

`mcp_add/remove/list` 是运行期管理入口，不是唯一配置方式。用户也可以手写 `workspace/mcp_servers.json` 并重启 Worker：

- `mcp_add` 校验配置、连接并列举工具、整批注册成功后原子保存配置，再刷新搜索索引和 Skill 状态。
- `mcp_remove` 删除持久化配置、注销该 Server 的工具并关闭 Client。
- `mcp_list` 返回连接状态、工具、脱敏配置和失败诊断。
- `mcp_servers.json` 已存在时作为持久化事实源；运行期间手工修改没有文件监听，需要重启生效。

持久化 env 只允许 `${ENV_NAME}` 引用，真实值从进程环境解析，避免 Secret 写入 JSON。单个 Server 启动失败只标为 unavailable，不阻塞其他 MCP 或核心 Worker。

## 10. 普通 MCP 副作用的真实边界

MCP Actor 不自动重放失败调用，但普通 MCP 工具没有飞书最终回复那样的持久化 Outbound Effect。如果 `send_email` 已在远端执行、响应返回前连接断开，MemoPilot 只能得到连接类 ToolObservation，无法证明邮件是否已经发出。ReAct 后续再次调用仍可能造成重复。

因此可以宣称“连接重建不会在传输层自动重放失败工具”，不能宣称“所有 MCP 副作用都进入 unknown 对账”或 exactly-once。`risk=external-side-effect` 也不能提供这个保证。不可幂等 MCP 应优先使用服务端幂等键或专用可靠适配器；不能重新添加只有状态字段、没有意图持久化、稳定操作键、远端凭证和人工对账的伪 Effect。

## 11. 重点追问

> [!IMPORTANT]
> **Manifest 是权限清单吗？**
>
> 不是。当前可选 Manifest 只描述插件名称、版本、说明和作者。此前没有形成真实沙箱的强制 capability manifest 已删除。

> [!IMPORTANT]
> **Event Handler 是实际函数，为什么启动时说“注册”？**
>
> 装饰器先记录元数据，PluginManager 再把函数登记到 EventBus；只有运行期间对应事件发生时才真正执行。Tool 的登记与执行也是同样的两阶段。Handler 先登记只是实现顺序，不代表先执行。

> [!IMPORTANT]
> **Config 和 KV 有什么区别？**
>
> Config 是作者默认值与用户覆盖的合并输入；KV 是插件自己维护的游标、计数和运行状态。KV 集中放在工作区状态目录，不写回插件源码。

> [!IMPORTANT]
> **`always=true` 写在 Catalog 中吗？**
>
> 不写。它位于 SKILL.md frontmatter，由 Runtime 用于自动注入全文；Catalog 只展示可发现信息和可用状态。

> [!IMPORTANT]
> **Drift Skill 和普通 Skill 是同一个东西吗？**
>
> 文件包装思想相同，但原型中属于不同目录、Loader、状态和执行链。MemoPilot 的 Drift 消费链属于阶段 6。

> [!IMPORTANT]
> **`mcp_servers.json` 必须由 `mcp_add` 生成吗？**
>
> 不必。可以手工维护并重启 Worker；`mcp_add` 的价值是运行期立即连接、校验、整批注册和持久化。

> [!IMPORTANT]
> **MCP 重连会把失败的副作用工具自动再执行吗？**
>
> Actor 不会自动重放，只重建连接并处理后续排队请求；但普通 MCP 也没有通用 unknown/Effect 对账，LLM 再次调用仍可能重复，必须如实说明边界。

## 12. 已验证与不可宣称范围

已有离线测试覆盖插件装载、四类扩展点、生命周期异常、配置/KV、EventBus、动态 Prompt、Skills、工具筛选/LRU、MCP 官方 SDK Actor、动态 Registry、启动装配和并发上下文；阶段实现记录中的最近一次完整离线验收为 328 个测试通过，并通过 Ruff、MyPy、差异检查和构建。学习期间没有调用真实模型、飞书或第三方 MCP API。

当前不能宣称插件安全沙箱、热重载、初始化失败的全部副作用原子回滚、MCP HTTP/SSE、普通 MCP exactly-once、阶段 6 Drift 执行或未经本轮授权的真实 MCP 端到端验证。

### 阶段 5 面试表述

> MemoPilot 以统一 ToolRegistry 和七阶段 Runtime 为核心，迁移原型的可信本地 Python 插件合同，支持 PhaseModule、类型化 EventBus、工具前置 Hook 和装饰器工具，并提供配置、工作区 KV 与启动诊断。大量插件和 MCP 工具通过常驻集合、tool_search 和会话 LRU 分层暴露，避免全量 Schema 挤占 Prompt。Skills 根据工具依赖生成 Catalog，并按 always 或显式选择注入完整工作流；本地 stdio MCP 使用官方 SDK 和单 Server Actor 管理会话，将远端能力适配成统一 ToolObservation。系统明确区分风险元数据与真实副作用保证，不把普通 MCP 错误夸大为 exactly-once。

---

# 跨阶段速查

## 1. 一条被动消息的目标链路

```text
飞书消息
→ operational.db：Inbox + Job + Outbox
→ Redis P0 Stream
→ Worker 获取会话 Lease 与 Fencing Epoch
→ RuntimeJobExecutor
→ 七阶段 AgentRuntime + ReAct
→ SQLite Steps
→ 最终 Run/Job/Outbound Effect
→ 飞书发送
→ SQLite 终态
→ Redis ACK 与清理
```

阶段 0—2 已完成到 Runtime 返回和 Job/Step 落库；飞书发送与完整 Outbound Effect 链路属于阶段 3。

## 2. 最重要的边界

```text
SQLite 是事实源，Redis 是协调层。
Lease 减少并发，Fencing 拒绝失权写入。
工具失败是 Observation，执行权失败不是。
审计日志说明发生了什么，Checkpoint 才能恢复执行。
Provider 处理厂商差异，ReAct 保持通用。
```

## 3. 当前不能宣称已经完成的能力

- Step/Message 安全 Checkpoint 精确恢复。
- 插件、Skills、MCP、主动唤醒和定时任务。
- 未经当次明确授权的任何真实外部 API 验证。

## 4. 后续阶段更新模板

每个新阶段在本文件追加：

1. 该阶段解决的问题。
2. 原型实际做法。
3. MemoPilot 实际做法。
4. 差异产生的需求依据。
5. 关键数据流和失败路径。
6. 收益、代价与适用边界。
7. 面试可复述版本。
8. 已实现、待实现和不可宣称事项。

---

## 13. 阶段 6：统一 AgentTick 主动链增量

### 13.1 本轮真正新增了什么

主动 Source 仍由固定 Tick 拉取并写入 Reservoir，但业务决策不再拆成 Alert、Context、Content 三套 LLM。三类数据先形成同一个静态快照，再进入统一 AgentTick ReAct，优先级是：

```text
Alert > Content > Context-fallback > Drift
```

- Alert：同 Tick 全部合并成一条，evidence 必须包含所有 Alert ID。
- Content：最多 5 条，正文并发预取但按需揭示；逐条检索兴趣记忆、分类并收尾。
- Context：保留原始字段，只补充 `_source`、本地时间和 `awake_prob`；只做背景或末级 fallback。
- Drift：无可推送结果且满足最短间隔时，投递 P3 后台 Skill Job。

### 13.2 Resolve 和 ACK

AgentTick 只生成业务决定，真正发送仍经过 Effect/Outbox。发送前先用来源 URL/标题/ID 生成 delivery key，再用 LLM 比较近期已确认主动消息是否语义重复。只有 Effect confirmed 后才消费待发送事件并记录 delivery。

Content ACK 保留旧原型的三档 TTL：cited 168 小时、interesting-but-uncited 24 小时、discarded 720 小时。发送失败不消费 interesting；去重命中使用 24 小时 post-guard ACK。Context 只本地消费，不调用 Source ACK。

### 13.3 重点问题

> [!IMPORTANT]
> **为什么不能是 Alert → Context → Content？**
>
> Context 是环境背景，不是与 Content 平级的新主题。把它提前并单独交给 LLM，会让低价值状态变化抢在真正内容前发送；旧原型因此把它放在末级 fallback。

> [!IMPORTANT]
> **为什么多条 Alert 要合并？**
>
> Alert 时效性高，但连续逐条 follow-up 会在一个 Tick 内轰炸用户。整批合并同时保留全部 evidence，兼顾及时性、可追踪和打扰成本。

> [!IMPORTANT]
> **Context fallback 现在是否自动开启？**
>
> 尚未。业务链已经支持，但 Wake Policy 仍是独立未决边界，生产默认关闭。当前不能宣称 Energy、AnyAction、Hazard、动态 Tick 或最终唤醒算法已经完成。

### 13.4 面试表述

> 我把主动系统拆成两层：外层用固定 Tick、Redis 优先级、SQLite Reservoir 和 Effect 保证调度、抢占与恢复；内层迁移旧原型的统一 AgentTick，由一个可追踪 ReAct 按 Alert、Content、Context fallback 做语义决策。这样业务优先级只有一处，工具轨迹可审计，发送与 ACK 又不会因为模型或进程重启丢失。

---

# 2026-07-28：统一 Redis AgentLoop 删除导向重构

## 原型与 MemoPilot 的取舍

原型由单一 AgentLoop 调 CoreRunner，进程内 MessageBus 足以满足本地即时对话。MemoPilot 需要 Redis 的 P0～P3、Pending 接管、Lease 和抢占，但这只要求替换统一入口的派发介质，不要求再建一套 BackgroundTaskLoop。

当前链路：

```text
Channel / Scheduler / Turn 后处理
→ RedisTaskQueue
→ AgentLoop（Lease、Fencing、Pending、抢占、ACK）
→ CoreRunner（一次 kind 分发）
→ passive / proactive / schedule / memory / drift
→ 可见结果经 MessagePushTool 发送
→ ACK
```

## 关键故障边界

- P0 与 stop 标记由同一 Lua 原子写入；重复入站不重复设置 stop。
- 发送失败不 ACK，消息留在 Pending；旧 Lease 消失后由新 consumer 接管。
- schedule/memory 被用户消息抢占时不做“ACK 后重发”，避免中间崩溃丢任务；proactive/drift 过期后可 ACK 丢弃。
- 被动 Turn 提交时校验 Fencing。提交后发送前崩溃，重投读取旧回复，不重跑模型、不重复 `TurnCommitted`。
- `TurnCommitted` 在本地提交后、外部发送前同步 fanout，用于插件统计；记忆维护仍由稳定 Redis task 承担，避免双副作用入口。
- `activity_version` 不能被 P0 优先级替代：它负责拒绝模型判断期间已经因新用户活动而过期的主动发送。

## 收益、代价与面试表述

收益是生产 Python 净减少 317 行，删除 MessageBus、GatewayService 和 BackgroundTaskLoop 三个生产文件，消费、抢占、恢复和 ACK 只有一处。代价是被动消息也依赖 Redis；Redis 暂时不可用时 Channel 无法进入执行链。第一版接受该边界，因为 Redis 已是后台任务和会话协调的必需依赖。

> 面试表述：我没有把 Redis 做成业务数据库，而是把所有工作统一成有优先级的派发项。AgentLoop 只负责执行权和恢复，CoreRunner 只负责领域分发，SQLite 只保存业务事实，MessagePush 只负责送达。这样既保留了 Redis Pending/Lease 的可靠性，也比“被动一条 Bus、后台一条 Worker”更短、更容易验证。

当前不能宣称外部渠道 exactly-once：发送成功后、ACK 前崩溃仍可能产生极低概率重复；稳定 provider UUID 只能在渠道支持时降低重复。

---

# 2026-07-28：原型运行时插件对齐

## 原型实际方案

原型把运行时行为拆成可信本地插件：`tool_loop_guard` 拒绝连续重复工具批次，`context_pressure` 在上下文压力过高时请求自然收尾，`citation` 负责内部引用协议，`observe` 将 Turn、检索和记忆写入事件异步落到本地 SQLite；`status_commands`、`setup_helper` 和 `plugin_undo` 在 BeforeTurn 短路模型，`meme` 用工作区 manifest、分类目录和回复尾部 `<meme:tag>` 选择本地图片。原型插件共享 PluginManager、生命周期模块、事件总线和发送工具，不为每个插件建立独立执行循环。

## MemoPilot 的实现与差异依据

MemoPilot 复用了现有 PhaseModule、EventBus、OperationalRepository 和 MessagePushTool：

- 推理安全插件直接适配当前 ReAct 与 Prompt 合同，不建立通用规则引擎。
- `observe` 只迁 writer、SQLite schema、retention 和事件映射，不迁 Dashboard、HTTP reader 或前端字段。
- `/memorystatus` 和 `/kvcache` 只展示当前数据库与 Runtime 确实拥有的事实，不补原型中已经失去数据来源的字段。
- `/chatid`、`/myid` 使用 MemoPilot 的自动私聊登记和 `[proactive] enabled = true` 配置说明，不保留其他渠道的旧文案。
- `meme` 保留原型完整表情协议、manifest mtime 缓存和分类选图，资源固定为工作区 `memes/`；媒体继续走现有发送链。
- `/undo` 通过 OperationalRepository 的公开 SQLite 事务删除当前会话最近一轮完整 user/assistant 消息并回退整理游标，不复制原型 SessionManager、私有缓存或 Undo service。

差异来自当前事实模型：MemoPilot 的已提炼记忆没有可靠的“单轮消息 → 记忆项”可逆关系，因此 `/undo` 明确保留已提炼记忆，不伪造跨 SQLite 库原子回滚。这个限制比按文本猜测删除记忆更安全，也更容易解释。

## 统一链路与事件时机

```text
Redis P0～P3
→ AgentLoop（Lease / Fencing / Pending / 抢占）
→ CoreRunner
→ Phase Pipeline / ReAct / 插件
→ SQLite 提交最终 Turn
→ TurnCommitted
→ observe writer
→ MessagePushTool（文本 / 文件 / 图片）
→ Redis ACK
```

`TurnCommitted` 只在首次本地提交成功后、外部发送前发出；同一 Redis Pending 重放复用已经持久化的回复，不重跑模型，也不重复发根事件。Observe 订阅 `TurnCommitted`、`RetrievalCompleted` 和 `MemoryWritten`，事件只由各业务根成功路径发出；失败检索或失败写入不伪装成成功观察记录。Observe 只消费事件，不反向读取 Repository 私有实现。

## 媒体稳定发送 ID 与日志边界

被动回复的稳定 `provider_uuid` 由消息业务键产生。每个媒体项再以该 base、媒体序号和路径通过 UUIDv5 派生独立 ID；base 缺失时用 channel、chat_id、正文和媒体列表形成稳定名称。同一 `OutboundDispatch` 重放得到相同媒体 ID，同批两个媒体得到不同 ID。生产装配把这个 ID 原样传给 `FeishuChannel.send_image/send_file`，不再在媒体 wrapper 中生成 UUIDv4。`operational_v9` 同时把 assistant 媒体集合持久化到 `messages.media_json`，因此文本提交后、图片发送前崩溃的 Pending 重放仍会恢复同一媒体集合；损坏 JSON 明确失败且不 ACK，不静默降级为空列表。

终端没有新增日志框架。新增日志仅为：

- AgentLoop start、stop，以及每个任务一条包含 kind、session 和截断 preview 的处理摘要；
- 每个插件加载完成一条 Plugin loaded；
- SchedulerService start、stop。

LLM 调用、工具开始/结果继续复用 ReAct 的现有日志，文本与媒体发送继续复用 MessagePushTool 的现有日志。Observe writer 不逐事件刷终端，也不输出完整 Prompt、Secret 或长正文。

## 收益、代价与面试表述

收益是原型中已经验证的安全、状态查看、观察和轻量交互能力进入唯一 Redis AgentLoop，没有第二条插件执行链；Observe、媒体和命令都能从同一 Turn 边界解释。代价是本地插件仍属于可信代码，Observe 是本地 SQLite 而非 Dashboard，Undo 只能撤销会话事实，不能撤回外部消息或已提炼记忆。

> 面试表述：我迁移插件时没有按目录复制，而是从原型真实装配和测试反查行为，再适配到 MemoPilot 的统一 Redis AgentLoop。插件只扩展 Phase、事件或发送边界；TurnCommitted 在首次提交后发出，Observe 不参与业务决策；媒体重放使用由业务回复 ID、序号和路径派生的稳定 UUID；Undo 只做当前 Schema 能原子保证的会话回滚，并把不可逆记忆边界明确告诉用户。

本阶段完整离线验证为 `505 passed`，Ruff、MyPy 和 `git diff --check` 通过。未调用真实模型、Embedding、飞书或远程 MCP API，因此不能宣称真实外部 API 或真实渠道端到端验收已经完成。
## 2026-07-29：原型式生命周期与上下文压力

原型的 after-step 是 `copy → collect_pre → fanout → collect_post → return`，插件通过
`PhaseFrame.slots` 条件性导出 early-stop 与 telemetry；MemoPilot 现在直接复用这一语义，
没有另造一套插件执行器。`context_pressure` 使用 JSON 消息载荷估算 token，在约 80% 阈值
时写入 early-stop，ReAct 消费该信号结束当前循环。

MemoPilot 保留自己的 Redis/Lease/SQLite 运行边界，因此只扩展 `ReActObserver.after_step`
传递工具执行后的完整 working 消息，并在适配层把 Frame 的 declared produces 视为可选导出，
而 Mapping 仍严格校验。这样既复制了原型的实际代码/协议，又避免上下文压力插件在正常低压
路径抛错。使用本机无持久化 Redis 完成离线验收，完整套件 `493 passed`；未启用真实模型或外部 API。

---

# 2026-07-30：统一 AgentLoop 与上下文压力插件合流

## 原型与 MemoPilot 的边界

原型用单个 AgentLoop、进程内 MessageBus 和 Phase/Event 插件完成即时对话；`context_pressure`
在工具结果加入工作消息后执行 after-step，并在上下文接近窗口约 80% 时请求自然收尾。
MemoPilot 复用这套 Runtime 与插件语义，但因后台恢复、优先级、Pending 接管和会话协调的真实需求，
把所有任务统一发布到 Redis，由唯一 AgentLoop 持有 Lease/Fencing 并消费；SQLite 仍只保存业务事实。

## 本次合流

统一 Redis AgentLoop 分支与 `main` 上较新的上下文压力生命周期提交采用普通合并，不重写历史。
冲突只在真实交叉点处理：保留统一任务入口、插件和媒体装配，同时保留原型式 PhaseFrame
条件性 slot、工具结果后的压力检查、运行时上下文窗口和明确的 early-stop 原因。针对工具结果的回归
先证明调用前未跨阈值，再用工具返回的大结果跨阈值，避免测试被工具参数本身提前触发。

收益是执行权、恢复和 ACK 只有一条链，而 Runtime 生命周期继续贴近原型；代价是被动消息也依赖
Redis 可用性。当前不能宣称终端输出逐字等同原型、真实飞书媒体端到端已验收，或已经验证多 Worker
吞吐量。合流后在 `main` 使用本机无持久化 Redis 完成完整离线回归：`512 passed`，Ruff、MyPy 和
`git diff --check` 全部通过，未调用真实模型、Embedding、飞书或远程 MCP API。

> 面试讲法：我没有为 Redis 再造第二套 Runtime，而是把它限制在统一派发、执行权和故障恢复边界；
> AgentLoop 决定谁执行，CoreRunner 决定执行什么，原型式 Phase/Event 插件扩展同一条 ReAct 链，
> SQLite 保存业务事实，MessagePush 负责送达。

---

# 2026-07-30：记忆上下文向原型收敛

原型以单个 `memory_window=40` 派生 20 条热历史、20 条 Consolidation 保留、10 条最少整理窗口和 10 条 Recent Turns。MemoPilot 采用同一派生关系，但仍保留 Redis AgentLoop、Lease/Fencing、Pending 和 Manifest：配置只公开 `memory_window`，派生值通过 `memory_history_limit`、`memory_consolidation_keep_count`、`memory_consolidation_min_new_messages` 与 `memory_recent_turn_count` 注入 Runtime 和 Consolidation。

原型会把 assistant 工具调用、每个 Observation 和最终回复按组恢复到下一轮 Prompt。MemoPilot 将该组写入 `operational_v10.messages.tool_chain_json`，损坏 JSON 明确失败，Pending 重放复用已持久化记录；历史展开仍按持久化业务消息窗口计数，并从 user 边界开始，避免半轮 assistant 上下文。工具结果超过 10,000 字符时保留首尾并加截断标记。Consolidation 只读取 user/assistant 最终文本，因此不会把原始工具输出写入 Markdown 或向量记忆。

收益是短期对话与整理窗口不再独立漂移，且工具型回合在重启或 Pending 接管后仍具有模型真实看到过的因果链。成本是 Prompt 消息数可能高于 20 条业务消息，以及需要对 JSON 迁移/解析失败保持显式错误边界。面试可表述为：我没有引入第二套记忆系统，而是复用原型的窗口与工具链合同，并在 MemoPilot 的 Redis/SQLite 一致性边界内补上稳定持久化和可恢复性。
