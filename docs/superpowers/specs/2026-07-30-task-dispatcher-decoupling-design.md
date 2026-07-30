# MemoPilot 任务分发与领域职责解耦设计

> 日期：2026-07-30  
> 状态：已批准  
> 范围：统一 AgentLoop 之后的任务分发、被动回复、记忆、主动/Drift 与定时执行边界

## 1. 背景与问题

MemoPilot 已将被动回复、记忆维护、主动判断、Drift 和定时执行统一发布到优先级任务队列，并由唯一 `AgentLoop` 管理 Lease、Fencing、Pending、抢占、心跳和 ACK。

统一入口之后，当前 `CoreRunner` 同时承担了两种不同职责：

1. 根据任务 `kind` 选择业务入口；
2. 亲自执行被动回复和定时任务的完整业务流程。

具体表现为：

- `passive.turn` 在 `CoreRunner` 内完成消息解析、历史读取、工具链展开、Runtime 调用、Turn 提交、事件发布和最终发送；
- `schedule.run` 在 `CoreRunner` 内完成执行状态迁移、模式判断、Runtime 调用、发送和终态提交；
- `proactive.tick` 返回字符串 `"drift"` 后，由 `CoreRunner` 再理解业务语义并调用独立 `DriftExecutor`；
- 记忆任务由自定义 `MemoryTaskRouter` 分发，命名与其他 `Loop / Executor / Service` 缺少统一语义；
- `BackgroundTask` 同时承载 P0 用户消息和后台任务，名称与实际用途不符。

这使 `CoreRunner` 既像路由器又像领域服务，无法只通过接口理解，也导致历史装载等职责落在错误层级。

## 2. 设计原则

1. 同职责采用原型实际生产代码中的名称与边界。
2. MemoPilot 因统一任务队列新增的组件使用通用任务语言，不把 Redis 写进业务组件名称。
3. Redis 只出现在 `RedisTaskQueue` 和 AgentLoop 的协调边界；业务任务不是“Redis 任务”。
4. 不为字面对齐增加空壳。没有独立职责的 `AgentCore` 和只有单一输入的窄 `CoreRunner` 不迁移。
5. 原本属于被动、记忆、主动或调度领域的操作归还对应领域组件。
6. 优先迁移原型已经验证的完整行为单元及测试，再适配 MemoPilot 的 Lease、Fencing、Pending 和 SQLite 事实边界。
7. 本次只重构职责和命名，不改变任务优先级、稳定任务 ID、ACK 时机、主动 `activity_version` 屏障或外部副作用策略。

## 3. 顶层结构

```text
RedisTaskQueue
→ AgentLoop
→ TaskDispatcher
   ├─ passive.turn
   │    → PassiveTurnPipeline
   │         → AgentRuntime
   ├─ memory.*
   │    → MemoryService
   ├─ proactive.tick / drift.run
   │    → ProactiveLoop
   └─ schedule.run
        → SchedulerService
```

### 3.1 RedisTaskQueue

只负责：

- 发布和读取优先级 Stream；
- Consumer Group、Pending、Claim 和 ACK；
- 队列幂等键与传输记录。

### 3.2 AgentLoop

继续负责：

- 从队列取得任务；
- 获取、续租和释放 Lease；
- Pending 接管；
- 用户抢占与停止信号；
- 调用 `TaskDispatcher`；
- 发布派生任务并在业务成功后 ACK。

AgentLoop 不包含领域任务分支。

### 3.3 TaskDispatcher

替代当前 `CoreRunner`。它只根据 `AgentTask.kind` 调用领域入口，并返回派生任务。

禁止在 TaskDispatcher 内：

- 查询或修改 SQLite；
- 调用 AgentRuntime；
- 构造 Prompt 或历史；
- 发送渠道消息；
- 解析记忆、主动或定时任务 payload 的业务字段；
- 理解 `"drift"` 等领域结果。

## 4. 任务合同

将 `BackgroundTask` 改名为 `AgentTask`。它表达系统要完成的业务任务：

```python
@dataclass(frozen=True, slots=True)
class AgentTask:
    task_id: str
    kind: str
    priority: int
    session_key: str
    payload: dict[str, object]
    created_at: datetime
```

`QueueMessage` 继续表达 Redis 投递记录，并保留 Stream message ID 等传输字段。AgentLoop 解析 payload 后构造或恢复 `AgentTask`，TaskDispatcher 不接触 Redis 专属字段。

## 5. 被动回复

### 5.1 入口

不迁移原型中只做代理和插件注册转发的 `AgentCore`，也不保留只有一个 Agent 输入类型的 `CoreRunner`。

```text
TaskDispatcher
→ PassiveTurnPipeline.run
```

### 5.2 PassiveTurnPipeline

负责一次被动回复的外层业务流程：

```text
解析并校验 InboundMessage
→ 记录用户活动
→ 检查 Turn 是否已经提交
→ 未提交时调用 AgentRuntime
→ 持久化最终回复、媒体和工具链
→ 首次提交时发布 TurnCommitted
→ 使用稳定 provider UUID 发送最终回复
→ 返回派生的记忆任务
```

Pending 重放发现 Turn 已提交时：

- 不重新调用 AgentRuntime；
- 不覆盖已保存的回复、媒体和工具链；
- 不重复发布 `TurnCommitted`；
- 使用原稳定发送 ID 重试外发。

### 5.3 历史与 BeforeTurn

当前 `CoreRunner` 中的 `list_recent_messages + expand_history` 移入 Runtime 的 BeforeTurn 上下文准备链。

生产 Runtime 注入基于 `OperationalRepository` 的 `SessionManager` 协议实现。被动 Turn 不再预先传入已组装历史，而由 BeforeTurn 从同一 Session 快照产生：

- 记忆预检索使用的历史；
- PromptRender 使用的模型历史；
- 上下文压力使用的消息序列。

历史窗口按业务消息计数，并由 `SessionManager` 恢复：

```text
user
assistant(tool_calls)
tool(observation)
assistant(final)
```

定时、主动和 Drift 继续使用显式后台 `TurnInput`，不自动装载被动会话历史。

### 5.4 两层边界

- `PassiveTurnPipeline`：外层业务事务，负责重放、提交、事件和发送；
- `AgentRuntime`：内层模型生命周期，负责 BeforeTurn、BeforeReasoning、Prompt、ReAct 和 AfterReasoning。

不复制第二套 Phase Pipeline。

## 6. MemoryService

用户确认保留单一记忆领域入口：

```text
TaskDispatcher
→ MemoryService
   ├─ memory.consolidate
   ├─ memory.vectorize
   ├─ memory.post_response
   ├─ memory.optimize
   └─ memory.reinforce
```

`MemoryService` 由当前 `MemoryTaskRouter` 演进而来，但名称不再强调路由实现。它负责：

- 校验每类记忆任务 payload；
- 建立 Fencing 检查和 fenced write；
- 调用 `ConsolidationService`、`VectorizationService`、`PostResponseMemoryWorker`、`MemoryOptimizer` 和记忆强化入口；
- 保持 Consolidation 完成后向量化、Manifest 和稳定来源 ID 的现有顺序。

TaskDispatcher 不理解任何记忆内部字段。

## 7. ProactiveLoop 与 Drift

原型的 `ProactiveLoop` 拥有主动周期和 Drift 业务流程。MemoPilot 恢复同一领域边界：

```text
proactive.tick
→ ProactiveLoop.execute_task
   → 主动判断
   → send / skip / drift
   → drift 时在 ProactiveLoop 内继续执行

drift.run
→ ProactiveLoop.execute_drift_task
```

删除调度包中的顶层 `DriftExecutor`。其已验证的 Skill 选择、工具 Registry、完成状态和 Fencing 代码迁回 proactive 包，由 ProactiveLoop 内部复用。

TaskDispatcher 不再解析主动结果，也不决定何时进入 Drift。

## 8. SchedulerService 与 ApplicationScheduler

### 8.1 SchedulerService

恢复原型中 Scheduler 领域服务的完整职责，同时保留 MemoPilot 的 SQLite 与队列边界：

- 创建、查询和取消计划；
- 扫描到期执行实例；
- 执行 `instant` 或 `agent` 模式；
- 更新 execution 的 running/succeeded/failed/cancelled；
- 发送定时结果。

`schedule.run` 直接交给 `SchedulerService.execute_task`。当前 `CoreRunner._run_schedule` 的逻辑迁入该服务。

### 8.2 ApplicationScheduler

当前 `SystemScheduler + scheduling.scheduler.SchedulerService` 合并为 `ApplicationScheduler`，只负责周期性生产任务：

- 触发 `MemoryMaintenanceScheduler`；
- 调用 `SchedulerService.scan_due`；
- 生成主动 Tick；
- 将产生的 AgentTask 发布到队列。

它不执行任何到期任务。

## 9. 装配

Bootstrap 先构建领域组件：

1. AgentRuntime 与 SessionManager；
2. PassiveTurnPipeline；
3. MemoryService；
4. ProactiveLoop；
5. SchedulerService；
6. TaskDispatcher；
7. AgentLoop 与 ApplicationScheduler。

同一个 `SchedulerService` 实例同时提供工具操作、到期扫描和消费执行，避免当前生产者与消费者分别构造同类服务。

## 10. 错误与恢复边界

- TaskDispatcher 只对未知 `kind` 抛错。
- 各领域服务负责自身 payload 校验和状态迁移。
- Lease/Fencing 由 AgentLoop 获取，由领域服务在写入与外发前校验。
- `DeliveryError` 继续阻止 ACK，任务留在 Pending。
- `StaleActivityError` 对主动和 Drift 继续 ACK 丢弃，不重放陈旧结果。
- 取消、失权和基础设施失败保持当前语义。
- 不新增真实外部 API 测试。

## 11. 测试与验收

### TaskDispatcher

- 每个 kind 只调用对应领域入口；
- Dispatcher 不持有 Repository、AgentRuntime 或 OutboundPort；
- 未知 kind 显式失败；
- 派生 AgentTask 原样返回。

### PassiveTurnPipeline

- 两轮 Provider 测试继续证明 BeforeTurn 恢复完整工具历史；
- Pending 重放不调用模型、不覆盖工具链、不重复事件；
- 发送失败不返回成功；
- Session key 不一致明确失败。

### MemoryService

- 五类任务均在 MemoryService 内部分发；
- Fencing、Manifest、向量化顺序和强化幂等保持不变；
- TaskDispatcher 不包含记忆 kind 的 payload 分支。

### ProactiveLoop

- 主动结果为 drift 时在 ProactiveLoop 内执行；
- 直接 `drift.run` 使用同一内部实现；
- TaskDispatcher 不检查 `"drift"`；
- activity_version、发送确认和 Drift 完成记录保持不变。

### Scheduler

- SchedulerService 完成 instant/agent 执行和状态收敛；
- ApplicationScheduler 只生产并发布任务；
- TaskDispatcher 不修改 schedule execution；
- 工具、扫描和执行使用同一领域服务合同。

### 全局

- 删除生产代码中的 `CoreRunner`、`AgentCore`、`MemoryTaskRouter`、`DriftExecutor` 和误导性的 `BackgroundTask` 名称；
- 不出现 `RedisMemory*`、`RedisScheduler*` 等基础设施侵入领域的命名；
- 现有任务优先级、Pending、Lease、Fencing、ACK 和稳定 ID 回归通过；
- 完整离线测试、Ruff、MyPy 与全区间 `git diff --check` 通过；
- 不调用真实模型、Embedding、飞书或远程 MCP。
