## Custom SubAgent Routing

You have the following custom subagents available via `task(subagent_type=...)`:

| SubAgent | Trigger | Call Pattern | Note |
|----------|---------|--------------|------|


**General rule**: If any subagent listed above can fulfill a user request, delegate to it immediately. Do NOT attempt to implement the same functionality yourself.

---

## Skill Routing

当用户输入包含以下关键词时，加载对应 skill：

| 关键词 | Skill | 说明 |
|--------|-------|------|
| `/akg_pr` | `akg-pr` | 基于当前分支 diff 生成 PR 描述文件 |
| `/akg_issue` | `akg-issue` | 辅助构建 Issue 描述文件（Bug/RFC/Task） |

---

## Agent Switch Suggestion

当用户请求**算子优化**或**算子融合分析**（生成算子、优化算子、替换算子、分析模型融合机会等），**不要自行处理**，建议切换到 `op-optimizer`：

> 这个任务由专门的 `op-optimizer` Agent 处理。
> 请通过 **Tab 键**或 **`/agents`** 命令切换到 `op-optimizer`，然后描述您的需求。
