# MemoPilot 记忆上下文向原型收敛设计

> 日期：2026-07-30  
> 状态：已批准  
> 范围：短期历史、工具链历史、Consolidation 窗口配置

## 1. 目标

将 MemoPilot 的短期会话与 Consolidation 合同对齐到原型当前生产装配，同时保留 MemoPilot 已实现的统一 Redis AgentLoop、Lease、Fencing、Pending 接管、稳定任务 ID 和 Consolidation Manifest。

本次解决三个已确认差距：

1. MemoPilot 默认只向模型提供 12 条最终 user/assistant 消息，原型实际提供 20 条热消息。
2. MemoPilot 未把历史 Tool Call 与 Tool Result 重建到下一轮 Prompt，原型会持久化并完整重建工具链。
3. MemoPilot 的短期窗口、Consolidation 保留窗口和最小整理窗口独立配置，可能发生语义漂移；原型由一个 `memory_window` 派生统一比例。

## 2. 原型事实

原型 `config.toml` 使用 `memory_window=40`。生产 Bootstrap 通过 `MemoryConfig.keep_count` 和 `_memory_keep_count()` 将其换算为：

```text
keep_count = align_to_multiple_of_4(memory_window) // 2 = 20
min_new_messages = max(5, keep_count // 2) = 10
recent_turn_count = max(1, keep_count // 2) = 10
```

原型 `SessionStore` 在 `sessions.db/messages.tool_chain` 保存每轮工具调用分组。`Session.get_history()` 将每组恢复为：

```text
assistant(content, tool_calls)
tool(tool_call_id, truncated_result)
assistant(final_content)
```

工具结果使用 10,000 字符预算，超限时保留首尾并插入截断标记。

## 3. 配置合同

MemoPilot 新增唯一公开参数：

```python
memory_window: int = 40
```

派生属性固定为：

```text
aligned_window = max(4, ceil(memory_window / 4) * 4)
history_limit = aligned_window // 2
consolidation_keep_count = history_limit
consolidation_min_new_messages = max(5, history_limit // 2)
recent_turn_count = max(1, history_limit // 2)
```

删除以下三个公开配置字段：

```text
memory_short_term_message_limit
memory_consolidation_keep_count
memory_consolidation_min_new_messages
```

旧 TOML 若包含上述字段，由 Pydantic 的既有 extra 行为忽略；README、示例配置和测试统一只描述 `memory_window`。不增加双轨兼容层，避免继续维护两套配置来源。

默认 `memory_window=40` 时：

| 语义 | 数量 |
|---|---:|
| 普通回复原始热历史 | 20 条消息 |
| Consolidation 保留 | 20 条消息 |
| 完整 Consolidation 最小旧消息 | 10 条消息 |
| `RECENT_CONTEXT` Recent Turns | 10 条消息 |

## 4. 持久化合同

新增 `operational_v10.sql`：

```sql
ALTER TABLE messages
    ADD COLUMN tool_chain_json TEXT NOT NULL DEFAULT '[]';
```

工具链只写在 assistant 最终消息上；user 消息保持空数组。JSON 保存原型兼容的分组结构：

```json
[
  {
    "text": "模型本步的可选文本",
    "reasoning_content": "可选思考字段",
    "calls": [
      {
        "call_id": "call-1",
        "name": "list_dir",
        "arguments": {"path": "."},
        "result": "工具 Observation 文本"
      }
    ]
  }
]
```

只保存模型实际收到的最终参数与 Observation 文本，不保存 EventBus 私有对象、Python 类型或不可序列化结果。工具失败同样保存结构化 Observation 文本，因为它是后续模型曾看到的真实历史。

持久化 JSON 损坏时，历史读取必须明确抛出 `ValueError`，不得静默丢弃工具链。该行为与现有损坏 `media_json` 的失败边界一致。

Pending 重放若发现 Turn 已提交，继续复用已持久化的最终回复、媒体和工具链，不重新调用模型，也不覆盖原记录。

## 5. 工具链采集与历史重建

`CoreRunner` 从 `ReActResult.messages` 与 `tool_chain` 生成原型分组结构。分组以每条带 `tool_calls` 的 assistant 消息为边界，随后关联相同 `tool_call_id` 的 tool 消息；最后的自然语言回复继续存入 `messages.content`，不重复写入分组。

读取最近历史时，Repository 返回 user/assistant 业务消息及 assistant 的工具链。历史构造器按原型顺序展开：

1. user 消息转为 `ChatMessage.user`。
2. 每个 assistant 工具分组转为一条带 `FunctionCall` 的 `ChatMessage.assistant`。
3. 每个调用紧随一条 `ChatMessage.tool`；工具结果使用原型 10,000 字符首尾截断算法。
4. 最后追加 assistant 的最终回复。

窗口计数仍按持久化业务消息计算，而不是按展开后的 OpenAI 消息数计算。默认最近 20 条业务消息可能展开为更多 Prompt 消息，这是原型合同的一部分。

媒体继续沿用现有 `media_json`。本次不新增历史图片重新内联，也不扩展主动消息历史格式。

## 6. Consolidation 与 Recent Turns

`ConsolidationService` 接收由 `memory_window` 派生的 `keep_count=20` 与 `min_new_messages=10`。

每轮提交仍发布 `memory.consolidate`：

- 未达到完整整理阈值时，只刷新 Recent Turns。
- 达到阈值时，整理最旧的可归档窗口，同时刷新 Compression 和 Recent Turns。
- Recent Turns 数量不再在 Service 内隐式依赖 `keep_count // 2`，而由派生配置显式注入，确保配置合同可测试。

Consolidation 的输入继续只使用 user/assistant 最终文本，不把工具原始输出写入 Markdown 或向量记忆，避免把大段工具结果误当成用户事实。

## 7. 保留的 MemoPilot 边界

以下行为不回退到原型：

- 所有 Turn 和记忆维护继续经过 Redis P0～P3。
- AgentLoop 继续统一管理 Lease、Fencing、抢占、Pending 和 ACK。
- SQLite 继续使用 `operational.db`，不恢复独立 `sessions.db`。
- Consolidation 继续使用 Manifest 和稳定任务 ID。
- Turn 重放继续读取已提交结果，不重复调用模型或重复发布 `TurnCommitted`。

## 8. 测试与验收

### 配置

- 默认 `memory_window=40` 派生 `20/20/10/10`。
- 非 4 倍数向上对齐后再派生。
- Bootstrap 将同一派生合同传给 CoreRunner 与 Consolidation。

### 迁移与 Repository

- 全新数据库直接升级到 operational v10。
- v9 数据库升级后旧消息的 `tool_chain_json='[]'`，文本与媒体不变。
- 工具链可以往返保存和读取。
- 损坏 JSON 明确失败。
- Pending 重放不覆盖已提交工具链。

### 历史重建

- 单次调用恢复为 assistant tool_call → tool result → assistant final。
- 同一模型步骤的多个 Tool Call 保持同组、原顺序和 call ID。
- 多轮工具调用保持分组顺序。
- 成功与失败 Observation 都能恢复。
- 超过 10,000 字符的结果按原型保留首尾并带截断标记。
- 历史窗口按 20 条业务消息截取，并保持从 user 边界开始。

### Consolidation

- 29 条未整理消息只刷新最新 10 条 Recent Turns，不调用压缩模型。
- 30 条未整理消息整理旧 10 条、保留热 20 条并刷新最新 10 条。
- 工具链不会进入 Consolidation 对话文本。

### 完整验证

```powershell
uv run pytest -q
uv run ruff check .
uv run mypy
git diff --check
```

全程只使用 Fake Provider、临时 SQLite 和本地测试 Redis，不调用真实模型、Embedding、飞书或远端 MCP。

