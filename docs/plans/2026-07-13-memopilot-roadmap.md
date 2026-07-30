# MemoPilot 第一版实施路线图

> 技术合同、状态机和数据结构以 [`../design/2026-07-13-memopilot-design.md`](../design/2026-07-13-memopilot-design.md) 为准。本文件只控制实施顺序、阶段产物和验收门槛，不重复设计书细节。
>
> **2026-07-28 状态覆盖：** 阶段 1、3、7 中的 Inbox／Outbox、AgentJob／Run／Step、Outbound Effect、三进程和独立 BackgroundTaskLoop 已撤销。当前第一版采用单进程 Channel + Scheduler + 唯一 AgentLoop，所有任务通过 Redis P0～P3 进入同一执行入口；SQLite 只保存业务事实。后续验收与面试讲解均以该覆盖说明和设计书顶部覆盖说明为准。

> **2026-07-30 状态覆盖：** RedisTaskQueue 的唯一消费者 AgentLoop 仅负责 Lease、Fencing、Pending、抢占与 ACK；TaskDispatcher 只分流到 PassiveTurnPipeline、MemoryService、ProactiveLoop 和 SchedulerService。ApplicationScheduler 只生产任务，不执行到期任务。

## 目标

用两周完成可用于求职展示的 MemoPilot 第一版：支持飞书私聊、ReAct + Function Calling、分层记忆、主动唤醒、定时任务、插件、Skills、MCP stdio、Redis 协调和可查询的运行证据。

## 实施原则

- 在全新 Git 仓库中开发，项目名和 Python 包名统一为 `MemoPilot` / `memopilot`。
- 每一阶段先完成核心测试，再进入下一阶段；不在主链路未跑通时扩展 Stretch 功能。
- 先建立可靠任务和副作用边界，再接模型、记忆和主动能力。
- 每个阶段形成独立、可演示的提交，保留真实开发历史。
- 核心模块按已批准的技术合同实现，既有 FeishuChannel 只做新 Runtime 所需的接口适配。

## 两周阶段安排

| 阶段 | 时间 | 主要工作 | 阶段交付物 | 通过标准 |
|---|---:|---|---|---|
| 0. 新仓库与工程基线 | 第 1 天 | 初始化 uv/Python 3.12、目录、配置、测试框架和三库迁移 | 可安装的 `memopilot` 包、配置示例和空数据库迁移 | lint、类型检查和基础测试通过 |
| 1. 可靠任务底座 | 第 2–3 天 | operational.db、transactional inbox/outbox、P0–P3 Redis Streams、恢复扫描、lease 与 fencing | 入站事件可可靠生成 Job；Worker 可安全接管 | 覆盖 commit/XADD 崩溃窗口、重复投递、Redis 清空恢复、旧 Worker 失权 |
| 2. Agent Runtime 纵向链路 | 第 4–5 天 | Phase DAG、OpenAI-compatible Provider、DeepSeek、ReAct 循环、ToolRegistry、错误 observation、自然语言收尾 | Fake LLM 与 DeepSeek 均可完成一次含工具调用的 Turn | Phase 顺序可追踪；工具失败回到 LLM；达到上限后自然收尾 |
| 3. 飞书与外部副作用 | 第 5–6 天 | 接入长连接 FeishuChannel、删除 webhook、连接 inbox/outbox、稳定 UUID 与 outbound effect | 飞书私聊可以触发并收到完整 Agent 回复 | 重复事件不重复执行；用户活动可抢占；不明确发送不盲目重放 |
| 4. 分层记忆 | 第 6–8 天 | 短期消息、Markdown 长期记忆、sqlite-vec/关键词/RRF、自动预检索、`recall_memory`、Consolidation、Optimizer | 被动回复可使用并引用记忆；Turn 后异步归档 | 默认预检索无 HyDE；answer 双假设；部分文件崩溃可恢复；向量写入幂等 |
| 5. 插件、Skills 与 MCP | 第 8–9 天 | Python 插件 manifest/注册、Prompt/Phase/Hook 扩展、Skill 加载、MCP stdio 工具与 Wake Source 适配 | 外部 MCP 和本地插件无需改核心链路即可注册 | 注册冲突可诊断；失败插件不半注册；MCP 错误转 observation；ACK 合同可验证 |
| 6. 主动唤醒与定时任务 | 第 9–11 天 | 30 分钟固定 Tick、空闲准入、三类 Source Reservoir、统一 AgentTick ReAct、ACK/去重、Drift、`at/after/every`、`instant/agent` | 系统按 `Alert > Content > Context > Drift` 单路主动判断，并可执行后台 Skill 和用户定时任务 | 忙会话不创建主动 Job；ACK 可重放；定时实例不丢失且周期 coalesce |
| 7. 可观测性与部署 | 第 11–12 天 | diagnostic log、strategy trace、只读 Inspector、Health/Ready、Docker Compose、配置向导 | app/worker/scheduler/redis 一键启动，本地配置可一次生成，可从 Swagger 查看证据 | 日志脱敏；核心 Reader 可查询；容器重启后任务与记忆仍在；Secret 不进入 Git |
| 8. 故障验证与作品集收尾 | 第 13–14 天 | 集成测试、Wake 回放、关键 failpoint、真实 DeepSeek/飞书/MCP E2E、README、架构图、演示脚本 | 可发布的 v0.1.0 与面试演示材料 | 设计书第 24.3 节演示场景全部通过，Must 项无未完成项 |

阶段 2 与阶段 3 在第 5 天交汇：先用 Fake Channel 跑通 Runtime，再接真实飞书；不得让 SDK 调试阻塞核心 Agent Loop。阶段 4 完成前不开发主动算法，因为 Wake 的兴趣判断依赖稳定的 Turn embedding 和记忆检索。阶段 6 完成前不增加 Inspector 聚合，避免为未稳定的数据结构写展示层。

### 阶段 7 配置体验发现

- 本地使用者只编辑一个 `.env`；`config/default.yaml` 继续作为可提交、无 Secret 的公共默认值，不要求普通使用者同时维护多份配置。
- 增加 `uv run memopilot setup` 配置向导，参考原型 Setup Wizard，一次询问并写入 DeepSeek、Embedding、飞书和 Redis 配置。
- 向导从 `.env.example` 生成或增量更新 `.env`，不得把真实 API Key、飞书 Secret 或用户 ID 写入受 Git 跟踪的文件。
- 向导结束前执行必填项校验和最小连接检查，并清楚报告缺失配置；生产部署仍允许操作系统环境变量覆盖 `.env`。

## 里程碑

### M1：被动聊天可用（第 6 天结束）

- 飞书私聊能够进入 Redis P0。
- Runtime 能调用工具并返回自然语言。
- 同会话严格串行，重复事件和发送重试受幂等保护。

### M2：核心简历能力可用（第 11 天结束）

- 自动预检索、`recall_memory`、Consolidation 和向量写入完整运行。
- 插件、Skills 与 MCP stdio 能扩展工具和主动 Source。
- Wake、Drift 与用户定时任务都经过 Redis，并受用户消息抢占。

### M3：作品集可发布（第 14 天结束）

- Docker Compose 一键启动。
- 自动化测试覆盖设计中列出的高风险崩溃窗口。
- Inspector、README 和演示脚本能证明简历中的每一条描述。
- README、设计文档和演示脚本能够准确说明当前实现范围。

## 范围控制

两周内只做设计书第 23 节的 Must。以下内容发现时间不足时直接推迟：主动图片/文件推送的端到端幂等、插件热重载、高级 Drift 调参、完整 Dashboard、跨主机部署和复杂媒体 E2E。不得通过删除 Redis 抢占、记忆异步链、MCP、插件或定时任务来换取表面完成，因为这些是简历的核心证据。

## 开工门槛

- 设计规格通过复审，没有 P0/P1 可靠性缺口。
- 新仓库路径、项目名 `MemoPilot` 和包名 `memopilot` 确认。
- DeepSeek、Embedding、飞书和至少一个 MCP 的测试配置可用，但 Secret 不写入仓库。
- 第一阶段开始后按本路线图更新 `task_plan.md` 和 `progress.md`，技术细节回写设计书而不是堆进路线图。

## 2026-07-30 阶段状态

- 统一 Redis AgentLoop、原型运行时插件迁移与 `main` 上较新的上下文压力生命周期实现已经合流。
- 合流采用普通 merge 保留两侧历史；上下文压力在工具结果进入工作消息后判断，PhaseFrame 保留原型的条件性 slot 语义。
- `main` 在本机无持久化 Redis 下完成完整离线验收：`512 passed`，Ruff、MyPy、`git diff --check` 通过。
- 本阶段没有调用真实外部 API；真实飞书媒体端到端、终端逐字一致性和多 Worker 吞吐量仍不得宣称已验收。
