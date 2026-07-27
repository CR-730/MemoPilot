---
name: create-drift-skill
description: 在工作区 skills 下创建或更新一个后台 Skill，用于把新的长期小任务沉淀成可复用技能
background_allowed: true
---

# 创建 Drift Skill

## 目标

把适合反复执行的小任务沉淀到工作区 `skills/<skill_name>/SKILL.md`。

## 何时使用

- 发现有新的长期任务适合放进 Drift
- 现有后台 Skill 太旧，需要补充流程或 working files

## 工作流

1. 先判断当前上下文里是否已有明确可沉淀的长期任务。没有明确可沉淀的长期任务时，不要询问用户，也不要创建空模板；直接调用 `finish_drift` 并用 `message_result=silent` 收尾。
2. 确认目标 Skill 名是否明确，并检查 `skills/<skill_name>/` 是否已存在。
3. 读取已有 `SKILL.md`，如果已存在就在原基础上更新；不存在再创建。
4. `SKILL.md` 顶部 frontmatter 至少包含：

```text
---
name: <skill_name>
description: <一句话描述>
background_allowed: true
---
```

5. 正文只写完成当前任务真正需要的最小流程，避免空泛模板。

## 约束

- Skill 文件必须写到工作区 `skills/` 下，不要写到仓库内建目录
- 不要为了一个一次性动作创建 Skill
- 如果只是当前 Skill 的进展变化，优先更新它的 working files 或 state，而不是新建 Skill
- 结束流程必须写清 `finish_drift.message_result`：已成功推送写 `sent`，静默结束写 `silent`
