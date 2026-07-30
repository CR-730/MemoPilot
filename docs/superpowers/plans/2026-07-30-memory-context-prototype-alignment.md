# 记忆上下文原型对齐实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）逐任务实现此计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 将 MemoPilot 的热历史、工具链历史与 Consolidation 窗口合同对齐原型生产实现，同时保留统一 Redis AgentLoop 和现有可靠性边界。

**架构：** 用一个 `memory_window=40` 派生 `20/20/10/10` 四个窗口值；在 operational v10 为 assistant 消息增加 `tool_chain_json`，由独立历史编解码模块把 ReAct working messages 转成原型分组并在下一轮展开为 assistant tool_calls、tool observations 和最终回答。Consolidation 仍只消费 user/assistant 最终文本。

**技术栈：** Python 3.12、Pydantic Settings、SQLite、Redis Streams、pytest、uv、Ruff、MyPy

---

## 文件结构

- 创建 `src/memopilot/runtime/history.py`：原型兼容的工具链序列化、历史展开与 10,000 字符工具结果截断。
- 创建 `src/memopilot/persistence/schema/operational_v10.sql`：为业务消息增加 `tool_chain_json`。
- 修改 `src/memopilot/config.py`：用 `memory_window` 统一派生四个窗口值。
- 修改 `src/memopilot/bootstrap.py`：把同一派生配置注入 CoreRunner 与 Consolidation。
- 修改 `src/memopilot/persistence/migrations.py`：注册 operational v10。
- 修改 `src/memopilot/tasks/operational.py`：工具链随 assistant 最终消息原子提交并往返读取。
- 修改 `src/memopilot/runtime/background.py`：提交 ReAct 工具历史并在下一轮重建完整历史。
- 修改 `src/memopilot/memory/consolidation.py`：显式使用派生的 Recent Turns 数量。
- 修改 `tests/unit/test_config.py`、`tests/unit/test_migrations.py`、`tests/unit/tasks/test_task_contract.py`、`tests/unit/runtime/test_background.py`、`tests/unit/memory/test_consolidation.py`：TDD 行为与迁移测试。
- 修改 `tests/integration/test_layered_memory_flow.py`：确认工具历史不进入 Consolidation 文本且记忆链仍运行。
- 修改 `README.md`、`docs/learning/memopilot-engineering-notes.md`、`调优日志.md`：同步公开合同、阶段知识与本地取舍记录。

### 任务 1：统一原型窗口配置

**文件：**
- 修改：`src/memopilot/config.py`
- 修改：`src/memopilot/bootstrap.py`
- 测试：`tests/unit/test_config.py`
- 测试：`tests/unit/test_bootstrap.py`

- [ ] **步骤 1：编写默认派生合同的失败测试**

在 `tests/unit/test_config.py` 将旧三个字段断言替换为：

```python
def test_memory_window_matches_prototype_contract() -> None:
    settings = MemoPilotSettings()
    assert settings.memory_window == 40
    assert settings.memory_history_limit == 20
    assert settings.memory_consolidation_keep_count == 20
    assert settings.memory_consolidation_min_new_messages == 10
    assert settings.memory_recent_turn_count == 10
```

增加非 4 倍数测试：

```python
def test_memory_window_aligns_up_before_deriving_limits() -> None:
    settings = MemoPilotSettings(memory_window=41)
    assert settings.memory_history_limit == 22
    assert settings.memory_consolidation_keep_count == 22
    assert settings.memory_consolidation_min_new_messages == 11
    assert settings.memory_recent_turn_count == 11
```

- [ ] **步骤 2：运行配置测试确认红灯**

运行：

```powershell
uv run pytest -q tests/unit/test_config.py
```

预期：FAIL，提示 `memory_window` 或派生属性不存在。

- [ ] **步骤 3：实现唯一配置和派生属性**

在 `MemoPilotSettings` 中删除三个旧字段，增加：

```python
memory_window: int = Field(default=40, gt=0)

@property
def memory_history_limit(self) -> int:
    aligned = max(4, ((self.memory_window + 3) // 4) * 4)
    return aligned // 2

@property
def memory_consolidation_keep_count(self) -> int:
    return self.memory_history_limit

@property
def memory_consolidation_min_new_messages(self) -> int:
    return max(5, self.memory_history_limit // 2)

@property
def memory_recent_turn_count(self) -> int:
    return max(1, self.memory_history_limit // 2)
```

Bootstrap 对 CoreRunner 使用 `settings.memory_history_limit`，对 Consolidation 使用其余三个派生属性。

- [ ] **步骤 4：补 Bootstrap 注入断言并确认绿灯**

在 `tests/unit/test_bootstrap.py` 的 Runtime 装配测试中断言：

```python
assert bundle.core_runner.short_term_message_limit == 20
assert bundle.memory_tasks.consolidation.keep_count == 20
assert bundle.memory_tasks.consolidation.min_new_messages == 10
assert bundle.memory_tasks.consolidation.recent_turn_count == 10
```

运行：

```powershell
uv run pytest -q tests/unit/test_config.py tests/unit/test_bootstrap.py
```

预期：PASS。

- [ ] **步骤 5：提交配置合同**

```powershell
git add src/memopilot/config.py src/memopilot/bootstrap.py tests/unit/test_config.py tests/unit/test_bootstrap.py
git commit -m "重构：对齐原型记忆窗口配置"
```

### 任务 2：为工具链历史增加 v10 持久化

**文件：**
- 创建：`src/memopilot/persistence/schema/operational_v10.sql`
- 修改：`src/memopilot/persistence/migrations.py`
- 修改：`src/memopilot/tasks/operational.py`
- 测试：`tests/unit/test_migrations.py`
- 测试：`tests/unit/tasks/test_task_contract.py`

- [ ] **步骤 1：编写 v9→v10 迁移红灯测试**

在 `tests/unit/test_migrations.py` 建立 v9 数据库并插入一轮带媒体的旧消息，升级后断言：

```python
columns = {
    row["name"]
    for row in connection.execute("PRAGMA table_info(messages)").fetchall()
}
assert "tool_chain_json" in columns
rows = connection.execute(
    "SELECT role, content, media_json, tool_chain_json "
    "FROM messages ORDER BY session_position"
).fetchall()
assert [row["tool_chain_json"] for row in rows] == ["[]", "[]"]
```

- [ ] **步骤 2：运行迁移测试确认红灯**

```powershell
uv run pytest -q tests/unit/test_migrations.py
```

预期：FAIL，operational 目标版本仍为 9 或列不存在。

- [ ] **步骤 3：实现 operational v10**

创建：

```sql
ALTER TABLE messages
    ADD COLUMN tool_chain_json TEXT NOT NULL DEFAULT '[]';
```

并将 `DatabaseKind.OPERATIONAL` 版本序列扩展到 10。

- [ ] **步骤 4：编写 Repository 往返与损坏 JSON 红灯测试**

在 `tests/unit/tasks/test_task_contract.py`：

```python
tool_chain = (
    {
        "text": "",
        "calls": [
            {
                "call_id": "c1",
                "name": "list_dir",
                "arguments": {"path": "."},
                "result": '{"ok":true}',
            }
        ],
    },
)
committed = repository.commit_turn(
    message,
    assistant_content="完成",
    assistant_tool_chain=tool_chain,
)
records = repository.list_recent_messages(message.session_key, limit=20)
assert records[-1].tool_chain == tool_chain
```

再把 assistant 的 `tool_chain_json` 更新为损坏文本，断言 `list_recent_messages` 抛出匹配 `tool_chain_json` 的 `ValueError`。

- [ ] **步骤 5：实现原子提交与读取**

扩展：

```python
class MessageRecord:
    ...
    tool_chain: tuple[dict[str, object], ...] = ()
```

`commit_turn()` 增加：

```python
assistant_tool_chain: tuple[dict[str, object], ...] = ()
```

写入时对 user 使用 `[]`，对 assistant 使用 `json.dumps(..., ensure_ascii=False)`；重放时校验新传工具链与持久化值一致。`list_recent_messages()` 解析并验证顶层必须是 list、元素必须是对象。

- [ ] **步骤 6：运行迁移与 Repository 测试确认绿灯**

```powershell
uv run pytest -q tests/unit/test_migrations.py tests/unit/tasks/test_task_contract.py
```

预期：PASS。

- [ ] **步骤 7：提交持久化合同**

```powershell
git add src/memopilot/persistence/schema/operational_v10.sql src/memopilot/persistence/migrations.py src/memopilot/tasks/operational.py tests/unit/test_migrations.py tests/unit/tasks/test_task_contract.py
git commit -m "迁移：持久化会话工具调用历史"
```

### 任务 3：按原型重建完整工具历史

**文件：**
- 创建：`src/memopilot/runtime/history.py`
- 修改：`src/memopilot/runtime/background.py`
- 测试：`tests/unit/runtime/test_background.py`

- [ ] **步骤 1：编写历史序列化和展开红灯测试**

覆盖以下场景：

```python
def test_round_trips_grouped_tool_history() -> None:
    working = (
        ChatMessage.user("查目录"),
        ChatMessage.assistant(
            content="我先检查",
            tool_calls=(
                FunctionCall("c1", "list_dir", {"path": "."}),
                FunctionCall("c2", "read_file", {"path": "README.md"}),
            ),
        ),
        ChatMessage.tool(call_id="c1", name="list_dir", content="a.py"),
        ChatMessage.tool(call_id="c2", name="read_file", content="# MemoPilot"),
        ChatMessage.assistant(content="检查完成"),
    )
    groups = serialize_tool_history(working)
    history = expand_business_history(
        (
            MessageRecord.user(...),
            MessageRecord.assistant(..., content="检查完成", tool_chain=groups),
        )
    )
    assert [message.role for message in history] == [
        "user", "assistant", "tool", "tool", "assistant"
    ]
    assert [call.id for call in history[1].tool_calls] == ["c1", "c2"]
```

另测：

- 多轮 assistant tool-call 分组顺序；
- 失败 Observation 文本原样恢复；
- 历史以 leading user 边界开始；
- 10,001 字符工具结果产生首尾截断标记；
- 空工具链仍只生成 user/assistant。

- [ ] **步骤 2：运行目标测试确认红灯**

```powershell
uv run pytest -q tests/unit/runtime/test_background.py
```

预期：FAIL，历史编解码函数或工具链提交参数不存在。

- [ ] **步骤 3：实现专用历史模块**

`runtime/history.py` 提供：

```python
TOOL_RESULT_CHAR_BUDGET = 10_000

def serialize_tool_history(
    messages: Sequence[ChatMessage],
) -> tuple[dict[str, object], ...]: ...

def expand_business_history(
    records: Sequence[MessageRecord],
) -> tuple[ChatMessage, ...]: ...

def truncate_tool_result(content: object) -> str: ...
```

序列化只收集带 `tool_calls` 的 assistant 消息及紧随其后的匹配 tool 消息；若 call ID 缺少结果则写入空字符串，保持原模型调用事实。展开时构造 `FunctionCall` 和 `ChatMessage.tool`，最终追加业务 assistant 文本。

- [ ] **步骤 4：接入 CoreRunner**

首次执行：

```python
history = expand_business_history(
    repository.list_recent_messages(
        message.session_key,
        limit=self.short_term_message_limit,
    )
)
```

Runtime 返回后：

```python
assistant_tool_chain = serialize_tool_history(result.react.messages)
repository.commit_turn(
    ...,
    assistant_tool_chain=assistant_tool_chain,
)
```

重放分支继续使用 Repository 中的既有内容，不重新运行 Runtime。

- [ ] **步骤 5：加入 CoreRunner 集成断言并确认绿灯**

测试两轮被动 Turn：第一轮 Fake Provider 调用工具，第二轮捕获 Provider 输入并断言历史中存在：

```text
user → assistant(tool_calls) → tool → assistant(final) → current user
```

同时断言 Pending 重放不会再次调用 Provider，也不会覆盖 `tool_chain_json`。

运行：

```powershell
uv run pytest -q tests/unit/runtime/test_background.py tests/unit/runtime/test_engine.py tests/unit/runtime/test_react.py
```

预期：PASS。

- [ ] **步骤 6：提交历史重建**

```powershell
git add src/memopilot/runtime/history.py src/memopilot/runtime/background.py tests/unit/runtime/test_background.py
git commit -m "重构：恢复原型完整工具历史"
```

### 任务 4：对齐 Consolidation、文档与整体行为

**文件：**
- 修改：`src/memopilot/memory/consolidation.py`
- 修改：`tests/unit/memory/test_consolidation.py`
- 修改：`tests/integration/test_layered_memory_flow.py`
- 修改：`README.md`
- 修改：`docs/learning/memopilot-engineering-notes.md`
- 修改：`调优日志.md`

- [ ] **步骤 1：编写 29/30 消息窗口红灯测试**

用参数：

```python
service = ConsolidationService(
    database,
    markdown,
    extractor,
    recent_context=recent_context,
    keep_count=20,
    min_new_messages=10,
    recent_turn_count=10,
)
```

断言：

- 29 条未整理消息不调用 extractor，只刷新最新 10 条 Recent Turns；
- 30 条未整理消息调用一次 extractor，整理最旧 10 条；
- `last_consolidated_position == 10`；
- Compression 输入不包含工具结果文本。

- [ ] **步骤 2：运行测试确认红灯**

```powershell
uv run pytest -q tests/unit/memory/test_consolidation.py tests/integration/test_layered_memory_flow.py
```

预期：FAIL，`recent_turn_count` 参数不存在或仍使用 `keep_count // 2` 隐式计算。

- [ ] **步骤 3：实现显式 Recent Turns 合同**

构造器增加：

```python
recent_turn_count: int
```

校验其大于 0，并在 `_recent_turns()` 中使用：

```python
recent_count = min(self.recent_turn_count, self.keep_count)
```

Consolidation 继续从 `messages.role/content` 构建文本，不读取 `tool_chain_json`。

- [ ] **步骤 4：运行记忆、运行时和迁移相关回归**

```powershell
uv run pytest -q tests/unit/memory tests/unit/runtime tests/unit/tasks tests/unit/test_config.py tests/unit/test_bootstrap.py tests/unit/test_migrations.py tests/integration/test_layered_memory_flow.py
```

预期：PASS。

- [ ] **步骤 5：更新公开说明、工程学习手册与调优日志**

README 写明：

```text
memory_window=40 派生 20 条热历史、20 条整理保留、
10 条最小整理窗口和 10 条 Recent Turns；
历史 Tool Call/Tool Result 会持久化并在后续 Turn 中恢复。
```

工程手册记录：

- 原型实际的 `40→20/10/10`；
- MemoPilot 旧 `12/5/6` 的差距；
- 本次对齐原因、Prompt 成本与历史完整性的权衡；
- Redis/Fencing/Manifest 仍是 MemoPilot 的可靠性增量；
- 不宣称已完成长期对话质量最优参数实验。

调优日志按长期模板记录发现方式、根因、原型对照、方案、验证证据、面试讲法和状态。

- [ ] **步骤 6：执行完整离线验收**

确认测试不会读取真实 API 配置，然后运行：

```powershell
uv run pytest -q
uv run ruff check .
uv run mypy
git diff --check
```

验收标准：

- 全部命令退出码为 0；
- 未调用真实模型、Embedding、飞书或远端 MCP；
- `git status --short` 只包含本阶段预期文件；
- operational v9→v10 保留旧消息与媒体；
- 两轮 Fake Provider 场景证明工具历史真实进入下一轮；
- 29/30 消息场景证明窗口边界。

- [ ] **步骤 7：提交阶段文档与最终修正**

```powershell
git add src/memopilot/memory/consolidation.py tests/unit/memory/test_consolidation.py tests/integration/test_layered_memory_flow.py README.md docs/learning/memopilot-engineering-notes.md
git commit -m "文档：记录记忆上下文原型对齐"
```

`调优日志.md` 只保留本机，不加入公开提交。

