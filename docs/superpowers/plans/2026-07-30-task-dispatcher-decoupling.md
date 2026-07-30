# MemoPilot 任务分发职责解耦实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 用纯 `TaskDispatcher` 替换职责膨胀的 `CoreRunner`，将被动回复、记忆、主动/Drift 和定时执行归还对应领域组件，并恢复原型中可复用的被动管线、主动循环和调度职责。

**架构：** `RedisTaskQueue` 只传输，`AgentLoop` 只协调执行权，`TaskDispatcher` 只按 `AgentTask.kind` 分流。被动回复进入 `PassiveTurnPipeline`，记忆五类任务进入 `MemoryService`，主动与 Drift 进入 `ProactiveLoop`，定时执行进入 `SchedulerService`；历史恢复由 Runtime BeforeTurn 通过 SessionManager 端口完成。

**技术栈：** Python 3.12、asyncio、Redis Streams、SQLite、pytest、uv、Ruff、MyPy。

---

## 文件结构

### 新建

- `src/memopilot/tasks/agent_task.py`：通用 `AgentTask` 业务任务合同。
- `src/memopilot/runtime/task_dispatcher.py`：只做 kind 到领域入口的分发。
- `src/memopilot/runtime/passive_turn.py`：被动 Turn 的重放、Runtime、提交、事件和发送管线。
- `src/memopilot/runtime/session.py`：基于 `OperationalRepository` 的 SessionManager/历史端口。
- `tests/unit/runtime/test_task_dispatcher.py`：纯分流合同。
- `tests/unit/runtime/test_passive_turn.py`：被动管线和 Pending 重放。
- `tests/unit/runtime/test_session.py`：BeforeTurn 历史恢复。

### 移动或删除

- 删除 `src/memopilot/runtime/background.py`，职责分别迁入 TaskDispatcher、PassiveTurnPipeline 和 SchedulerService。
- 删除 `src/memopilot/tasks/background.py`，调用方改用 `agent_task.py`。
- 删除 `src/memopilot/memory/tasks.py`，替换为 `src/memopilot/memory/service.py` 中的 `MemoryService`。
- 删除 `src/memopilot/scheduling/drift_executor.py`，行为迁入 proactive 包。
- 将原 `tests/unit/runtime/test_background.py` 的用例按职责拆入 dispatcher/passive/scheduler 测试。

### 修改

- `src/memopilot/runtime/agent_loop.py`：QueueMessage 转 AgentTask，并调用 TaskDispatcher。
- `src/memopilot/runtime/engine.py`：BeforeTurn 解析 Session 历史请求。
- `src/memopilot/runtime/history.py`：保留工具链转换，供 SessionManager 使用。
- `src/memopilot/memory/service.py`：MemoryService 内部分发五类任务。
- `src/memopilot/proactive/loop.py`：主动 Tick 内部接管 Drift。
- `src/memopilot/proactive/drift_runtime.py`：承载原 Drift 已验证的运行辅助。
- `src/memopilot/scheduling/service.py`：领域类统一为 SchedulerService，并接管 schedule.run。
- `src/memopilot/scheduling/scheduler.py`：收敛为 ApplicationScheduler，只生产任务。
- `src/memopilot/bootstrap.py`：按领域依赖顺序组装并共享实例。
- `src/memopilot/scheduling/contracts.py`、`repository.py`、`memory/scheduler.py`、`tasks/operational.py`、`tasks/redis_queue.py`、`runtime/common_tools/shell.py`：`BackgroundTask` 改为 `AgentTask`。
- 对应单元、集成和 Bootstrap 测试：更新名称和真实调用链断言。
- `README.md`、`docs/learning/memopilot-engineering-notes.md`、`调优日志.md`：同步最终职责与面试表述；后两者按本地排除规则只在本地维护，不强制加入 Git。

---

### 任务 1：建立 AgentTask 与纯 TaskDispatcher

**文件：**
- 创建：`src/memopilot/tasks/agent_task.py`
- 创建：`src/memopilot/runtime/task_dispatcher.py`
- 创建：`tests/unit/runtime/test_task_dispatcher.py`
- 修改：`src/memopilot/tasks/redis_queue.py`
- 修改：所有 `BackgroundTask` 生产方和测试

- [ ] **步骤 1：为名称和纯分流合同编写失败测试**

测试必须断言：

```python
task = AgentTask(
    task_id="t1",
    kind="passive.turn",
    priority=0,
    session_key="cli:c1",
    payload={"content": "hi"},
    created_at=NOW,
)
result = await dispatcher.dispatch(task, lease=LEASE, now=NOW)
assert passive.calls == [(task, LEASE)]
assert result == DERIVED_TASKS
assert not hasattr(dispatcher, "repository")
assert not hasattr(dispatcher, "runtime")
assert not hasattr(dispatcher, "outbound")
```

分别覆盖 `memory.*`、`proactive.tick`、`drift.run`、`schedule.run` 和未知 kind。未知 kind 预期 `ValueError("不支持的 Agent 任务")`。

- [ ] **步骤 2：运行红灯**

```powershell
uv run pytest -q tests/unit/runtime/test_task_dispatcher.py
```

预期：FAIL，`AgentTask` 和 `TaskDispatcher` 尚不存在。

- [ ] **步骤 3：实现最小合同**

`AgentTask` 使用规格中的七个字段。`TaskDispatcher` 构造函数只接收四个领域入口：

```python
TaskDispatcher(
    passive=passive_pipeline,
    memory=memory_service,
    proactive=proactive_loop,
    scheduler=scheduler_service,
)
```

`dispatch()` 只做 kind 选择和参数转交，不解析 payload。

- [ ] **步骤 4：保持生产入口暂不切换**

任务 1 只建立可独立验证的合同和纯 Dispatcher。AgentLoop 仍暂时调用现有 CoreRunner，避免在被动、记忆、主动和调度领域入口尚未完成前引入双轨兼容或破坏 MyPy；AgentLoop 到 TaskDispatcher 的原子切换统一放在任务 6。

- [ ] **步骤 5：机械迁移 BackgroundTask 名称**

把生产代码和测试中的 `BackgroundTask` 改为 `AgentTask`，删除旧别名，确保没有双轨名称：

```powershell
rg -n "from memopilot.tasks.background|class BackgroundTask" src tests
```

预期：无业务队列合同残余。`runtime/common_tools/shell.py` 中表示本地 Shell 子进程状态的私有 `_BackgroundTask` 不属于本次范围。

- [ ] **步骤 6：验证并提交**

```powershell
uv run pytest -q tests/unit/runtime/test_task_dispatcher.py tests/integration/test_redis_task_delivery.py
uv run ruff check src/memopilot/tasks src/memopilot/runtime tests/unit/runtime
uv run mypy
git diff --check
git add src tests
git commit -m "重构：建立统一任务分发合同"
```

---

### 任务 2：迁移 PassiveTurnPipeline 与 BeforeTurn Session 历史

**文件：**
- 创建：`src/memopilot/runtime/passive_turn.py`
- 创建：`src/memopilot/runtime/session.py`
- 创建：`tests/unit/runtime/test_passive_turn.py`
- 创建：`tests/unit/runtime/test_session.py`
- 修改：`src/memopilot/runtime/engine.py`
- 修改：`src/memopilot/runtime/history.py`
- 拆分：`tests/unit/runtime/test_background.py`

- [ ] **步骤 1：复制原型行为单元并写红灯**

实施前重新读取：

- 原型 `agent/core/passive_turn.py` 的 `PassiveTurnPipeline.run`；
- 原型 `agent/lifecycle/phases/before_turn.py` 的 AcquireSession/PrepareContext；
- 原型 `session/manager.py::Session.get_history`；
- MemoPilot 当前 `CoreRunner._run_passive` 及全部测试。

测试至少覆盖：

```text
首次 Turn：
  record activity → BeforeTurn history → Runtime → commit → TurnCommitted → dispatch

Pending 重放：
  persisted reply → no Runtime → no duplicate event → stable outbound ID
```

以及两轮 Provider 输入：

```text
user
assistant(tool_calls)
tool(success)
tool(failure)
assistant(tool_calls)
tool(result)
assistant(final)
current user
```

- [ ] **步骤 2：运行红灯**

```powershell
uv run pytest -q tests/unit/runtime/test_passive_turn.py tests/unit/runtime/test_session.py
```

预期：FAIL，当前仍由 CoreRunner 装载历史和执行被动流程。

- [ ] **步骤 3：实现 SessionManager 端口**

SessionManager 使用 OperationalRepository 读取业务消息，并调用现有 `expand_history()`；只返回标准 `ChatMessage`，不向 Runtime 暴露 SQLite Row。

为 BeforeTurn 定义类型化历史请求，而不是布尔开关：

```python
@dataclass(frozen=True, slots=True)
class SessionHistoryRequest:
    session_key: str
    limit: int
```

`TurnInput.history` 接受显式消息元组或 `SessionHistoryRequest`。BeforeTurn 遇到请求时通过 SessionManager 一次性解析，并把同一结果用于记忆预检索、Prompt 和上下文压力。

- [ ] **步骤 4：实现 PassiveTurnPipeline**

迁移当前 `_run_passive` 的完整已验证行为，不重写稳定 ID、提交事务或事件内容。Pipeline 构造 `TurnInput(history=SessionHistoryRequest(...))`，不直接调用 Repository 的历史查询。

- [ ] **步骤 5：验证历史只在 BeforeTurn 解析一次**

Fake SessionManager 记录调用次数：

```python
assert session_manager.calls == [("feishu:chat-1", 20)]
assert provider_history == expected_expanded_history
```

后台 `TurnInput` 显式传 `history=()` 时不得调用 SessionManager。

- [ ] **步骤 6：验证并提交**

```powershell
uv run pytest -q tests/unit/runtime/test_passive_turn.py tests/unit/runtime/test_session.py tests/unit/runtime/test_engine.py tests/unit/runtime/test_history.py
uv run ruff check src/memopilot/runtime tests/unit/runtime
uv run mypy
git diff --check
git add src/memopilot/runtime tests/unit/runtime
git commit -m "重构：恢复被动回复管线职责"
```

---

### 任务 3：将 MemoryTaskRouter 收敛为 MemoryService

**文件：**
- 创建：`src/memopilot/memory/service.py`
- 删除：`src/memopilot/memory/tasks.py`
- 创建或修改：`tests/unit/memory/test_service.py`
- 修改：`tests/integration/test_layered_memory_flow.py`
- 修改：`src/memopilot/bootstrap.py`

- [ ] **步骤 1：写 MemoryService 红灯**

逐一断言：

```text
memory.consolidate → consolidation.run → 有结果时 vectorization.run
memory.vectorize → vectorization.run
memory.post_response → post_response.run
memory.optimize → optimizer.run
memory.reinforce → reinforce_items_once
```

同时断言 Fencing 检查、fenced write、稳定 `usage_ref` 和缺字段错误保持现状。

- [ ] **步骤 2：运行红灯**

```powershell
uv run pytest -q tests/unit/memory/test_service.py tests/integration/test_layered_memory_flow.py
```

预期：FAIL，`MemoryService` 尚不存在。

- [ ] **步骤 3：最小迁移现有实现**

以当前 `MemoryTaskRouter` 为基线复制完整行为，只改类名、文件边界和接收 `AgentTask` 的接口；不得重新设计 Consolidation、向量化或强化顺序。

- [ ] **步骤 4：删除旧名并验证**

```powershell
rg -n "MemoryTaskRouter" src tests
```

预期：无输出。

- [ ] **步骤 5：提交**

```powershell
uv run pytest -q tests/unit/memory tests/integration/test_layered_memory_flow.py
uv run ruff check src/memopilot/memory tests/unit/memory
uv run mypy
git diff --check
git add src/memopilot/memory src/memopilot/bootstrap.py tests
git commit -m "重构：统一记忆服务入口"
```

---

### 任务 4：将 Drift 归还 ProactiveLoop

**文件：**
- 修改：`src/memopilot/proactive/loop.py`
- 修改：`src/memopilot/proactive/drift_runtime.py`
- 删除：`src/memopilot/scheduling/drift_executor.py`
- 修改：`tests/unit/proactive/test_loop.py`
- 移动或改写：`tests/unit/scheduling/test_drift_executor.py`
- 修改：`src/memopilot/bootstrap.py`

- [ ] **步骤 1：核对原型并写红灯**

重新读取原型：

- `bootstrap/proactive.py::build_proactive_runtime`；
- `proactive_v2/loop.py::ProactiveLoop`；
- `ProactiveTurnPipeline / AgentTick` 中进入 Drift 的实际代码与测试。

测试断言：

```python
await proactive.execute_task(proactive_tick, ...)
assert proactive_drift.calls == 1  # outcome=drift 时内部继续

await proactive.execute_task(drift_task, ...)
assert proactive_drift.calls == 1  # 直接 drift.run 复用同一实现
```

TaskDispatcher 测试必须证明它不读取 `"drift"` 返回值。

- [ ] **步骤 2：运行红灯**

```powershell
uv run pytest -q tests/unit/proactive/test_loop.py tests/unit/runtime/test_task_dispatcher.py
```

预期：FAIL，Drift 仍由 Dispatcher 外部二次调用。

- [ ] **步骤 3：迁移 Drift 已验证代码**

把 `DriftExecutor.execute_task` 的 Skill 选择、工具 Registry、Runtime、完成记录、Fencing 和异常收敛迁到 proactive 包。`ProactiveLoop` 根据 `AgentTask.kind` 或明确方法进入同一内部 Drift 路径。

不改变：

- `activity_version` 校验；
- `message_push` 发送方式；
- Drift Skill 工具白名单；
- 完成记录和取消/失败语义。

- [ ] **步骤 4：删除旧名并提交**

```powershell
rg -n "DriftExecutor|DriftTaskExecutor" src tests
```

预期：无输出。

```powershell
uv run pytest -q tests/unit/proactive tests/unit/runtime/test_task_dispatcher.py tests/unit/scheduling
uv run ruff check src/memopilot/proactive tests/unit/proactive
uv run mypy
git diff --check
git add src tests
git commit -m "重构：统一主动与漂移运行入口"
```

---

### 任务 5：恢复 SchedulerService 并建立 ApplicationScheduler

**文件：**
- 修改：`src/memopilot/scheduling/service.py`
- 修改：`src/memopilot/scheduling/scheduler.py`
- 修改：`tests/unit/scheduling/test_scheduler.py`
- 修改：`tests/unit/scheduling/test_tools.py`
- 修改：`src/memopilot/bootstrap.py`

- [ ] **步骤 1：对照原型并写执行红灯**

重新读取原型 `agent/scheduler.py::SchedulerService` 的 `_execute`、`_execute_and_reschedule` 与恢复测试。

对 MemoPilot 保留 SQLite/Redis 差异，测试：

```text
instant：running → outbound → succeeded
agent：running → AgentRuntime → outbound → succeeded
Runtime infrastructure_error → failed，不发送
DeliveryError → 保持可恢复状态并阻止 ACK
已终态 execution → 不重复运行
```

- [ ] **步骤 2：运行红灯**

```powershell
uv run pytest -q tests/unit/scheduling
```

预期：FAIL，执行逻辑仍在旧 CoreRunner。

- [ ] **步骤 3：扩展并改名领域 SchedulerService**

将当前 `ScheduleService` 改为 `SchedulerService`，保留 schedule/list/cancel/scan_due，并迁入 `_run_schedule` 的执行逻辑，公开：

```python
async def execute_task(
    self,
    task: AgentTask,
    *,
    lease: SessionLease,
    now: datetime,
) -> tuple[AgentTask, ...]:
    ...
```

- [ ] **步骤 4：合并生产循环为 ApplicationScheduler**

将当前 `SystemScheduler` 与外层发布 `SchedulerService` 收敛为 `ApplicationScheduler`。它只运行 Tick、收集 Memory/Schedule/Proactive 生产结果并发布，不执行到期任务。

- [ ] **步骤 5：确保单一 SchedulerService 实例**

Bootstrap 构建一个 SchedulerService，同时注入：

- schedule 工具；
- ApplicationScheduler 的 `scan_due`；
- TaskDispatcher 的 `schedule.run`。

- [ ] **步骤 6：删除旧名歧义并提交**

```powershell
rg -n "class ScheduleService|class SystemScheduler" src tests
```

预期：无输出。

```powershell
uv run pytest -q tests/unit/scheduling tests/unit/runtime/test_task_dispatcher.py tests/unit/test_bootstrap.py
uv run ruff check src/memopilot/scheduling tests/unit/scheduling
uv run mypy
git diff --check
git add src tests
git commit -m "重构：收敛定时任务领域服务"
```

---

### 任务 6：完成装配切换并删除 CoreRunner

**文件：**
- 修改：`src/memopilot/bootstrap.py`
- 修改：`src/memopilot/runtime/agent_loop.py`
- 删除：`src/memopilot/runtime/background.py`
- 修改：`tests/unit/test_bootstrap.py`
- 修改：`tests/integration/test_builtin_plugin_batch.py`
- 修改：所有残余导入

- [ ] **步骤 1：写生产装配红灯**

断言 RuntimeBundle/AppRuntime 中存在：

```text
task_dispatcher
passive_turn_pipeline
memory_service
proactive_loop
scheduler_service
application_scheduler
```

并断言 TaskDispatcher 不持有 Repository、AgentRuntime、OutboundPort。

- [ ] **步骤 2：运行红灯**

```powershell
uv run pytest -q tests/unit/test_bootstrap.py tests/integration/test_builtin_plugin_batch.py
```

预期：FAIL，Bootstrap 仍导出 CoreRunner 和旧服务。

- [ ] **步骤 3：切换 Bootstrap**

按规格第 9 节顺序组装，共享 SchedulerService 和 SessionManager。AgentLoop 在本任务中才从 QueueMessage 恢复 AgentTask，并将构造参数由 `runner` 原子改为 `dispatcher`；不得用 CoreRunner 实现 Dispatcher 协议，也不得保留双轨兼容。

- [ ] **步骤 4：删除旧代码和旧命名**

```powershell
rg -n "CoreRunner|AgentCore|MemoryTaskRouter|DriftExecutor|BackgroundTask|SystemScheduler" src tests
```

预期：无输出。原型仓库不在此检查范围。

- [ ] **步骤 5：运行跨领域定向回归并提交**

```powershell
uv run pytest -q tests/unit/runtime tests/unit/memory tests/unit/proactive tests/unit/scheduling tests/unit/test_bootstrap.py tests/integration/test_builtin_plugin_batch.py tests/integration/test_layered_memory_flow.py tests/integration/test_redis_task_delivery.py
uv run ruff check .
uv run mypy
git diff --check
git add src tests
git commit -m "重构：完成任务分发职责解耦"
```

---

### 任务 7：文档、学习记录与完整离线验收

**文件：**
- 修改：`README.md`
- 本地修改：`docs/learning/memopilot-engineering-notes.md`
- 本地修改：`调优日志.md`

- [ ] **步骤 1：同步当前架构**

README 只描述：

```text
RedisTaskQueue → AgentLoop → TaskDispatcher → 领域入口
```

说明：

- AgentLoop 管执行权；
- TaskDispatcher 只分流；
- PassiveTurnPipeline、MemoryService、ProactiveLoop、SchedulerService 各自拥有业务流程；
- ApplicationScheduler 只生产任务。

- [ ] **步骤 2：记录面试表述**

本地学习手册和调优日志记录：

- 问题如何被发现；
- 原 CoreRunner 为什么职责膨胀；
- 为什么不机械复制 AgentCore/CoreRunner 空壳；
- 为什么 MemoryService 保留内部五类分发；
- Redis 作为传输协调而不是领域身份；
- 改造收益、代价和不可宣称能力。

- [ ] **步骤 3：检查测试不会调用真实服务**

显式清空模型、Embedding、飞书相关环境变量，确认测试只使用 Fake Provider、临时 SQLite 和本地测试 Redis。

- [ ] **步骤 4：完整离线验证**

```powershell
uv run pytest -q
uv run ruff check .
uv run mypy
git diff --check c7031ac..HEAD
git status --short
```

预期：

- 完整测试全绿；
- Ruff、MyPy、全实施区间 diff-check 全绿；
- 工作区除本地忽略的学习手册与调优日志外无未提交改动；
- 未调用真实外部 API。

- [ ] **步骤 5：提交公开文档**

```powershell
git add README.md
git commit -m "文档：更新任务分发架构说明"
```

不要强制加入本地排除的学习手册或调优日志。

---

## 计划自检结果

- 规格中的任务合同、被动回复、MemoryService、ProactiveLoop/Drift、Scheduler、装配、错误恢复和测试要求均有对应任务。
- 计划中的接口均已明确，不保留双轨兼容层。
- `AgentTask`、`TaskDispatcher`、`PassiveTurnPipeline`、`MemoryService`、`ProactiveLoop`、`SchedulerService` 和 `ApplicationScheduler` 在所有任务中保持同名。
- 不引入 CoreRunner、AgentCore、Redis 前缀领域服务或新的执行循环。
- 每个阶段先红灯、后最小迁移、再定向验证并独立提交。
