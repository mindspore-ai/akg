## Custom SubAgent Routing

You have the following custom subagents available via `task(subagent_type=...)`:

| SubAgent | Trigger | Call Pattern | Note |
|----------|---------|--------------|------|


**General rule**: If any subagent listed above can fulfill a user request, delegate to it immediately. Do NOT attempt to implement the same functionality yourself.

---

## Agent Switch Suggestion

当用户请求**算子优化**或**算子融合分析**（生成算子、优化算子、替换算子、分析模型融合机会等），**不要自行处理**，建议切换到 `op-optimizer`：

> 这个任务由专门的 `op-optimizer` Agent 处理。
> 请通过 **Tab 键**或 **`/agents`** 命令切换到 `op-optimizer`，然后描述您的需求。
