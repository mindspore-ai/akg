[English Version](../DebugSaveResume.md)

# Debug Save/Resume 设计

## 1. 背景

AKG Agents 的算子生成流程基于 LangGraph 编排。一次任务可能包含设计、代码生成、静态检查、验证、性能分析和多轮修复。失败定位时仅依赖最终日志不够，常见诉求是：

- 中断后从已完成节点继续执行，避免重复调用 LLM 和 Verifier
- 固化某次失败现场，复现同一条状态路径
- 保持和 LangGraph 原生 save/resume 语义一致，避免自定义一套不可维护的状态机

因此 Debug 模块采用“两层职责”：

- LangGraph checkpoint：负责 workflow state 的 save/resume
- AKG Trace / WorkflowLogger / ReplayGuard：负责日志、节点输出、LLM 调用记录和回放辅助

## 2. 设计目标

- 默认关闭，不影响现有任务性能与行为
- 支持 `LangGraphTask`、KernelAgent 的 workflow tool 调用路径，以及 `OpTaskBuilderWorkflow`
- 使用 LangGraph 原生 checkpointer 与 `configurable.thread_id`
- 提供本地文件持久化实现，开发环境不强依赖数据库组件
- resume 时必须显式确认已有 checkpoint，不允许静默从空状态重跑

## 3. 配置

推荐配置：

```yaml
debug:
  enabled: true
  resume: false
  session_id: "relu-debug-001"
  checkpoint_dir: "~/.akg/langgraph_checkpoints"
```

从 checkpoint 继续：

```yaml
debug:
  enabled: true
  resume: true
  session_id: "relu-debug-001"
  checkpoint_dir: "~/.akg/langgraph_checkpoints"
```

也支持扁平字段，便于 CLI 或单测注入：

```yaml
debug_enabled: true
debug_resume: false
debug_session_id: "relu-debug-001"
debug_checkpoint_dir: "~/.akg/langgraph_checkpoints"
```

可选字段：

| 字段 | 说明 |
|---|---|
| `debug.enabled` | 开启 checkpoint 保存 |
| `debug.resume` | 从已有 checkpoint 继续 |
| `debug.session_id` / `debug.thread_id` | LangGraph `thread_id`，同一值用于 save/resume |
| `debug.checkpoint_dir` | checkpoint 文件目录 |
| `debug.checkpoint_file` | 显式指定单个 checkpoint 文件 |

默认文件名为：

```text
{checkpoint_dir}/{workflow_name}_{thread_id}.pkl
```

## 4. Save 流程

1. `BaseWorkflow.compile()` 检查 `debug.enabled/debug.resume`
2. 开启时创建 `FileCheckpointSaver`
3. `StateGraph.compile(checkpointer=..., debug=True)` 使用 LangGraph 原生 checkpoint 机制
4. 执行入口通过 `build_invoke_config()` 注入：

```python
{
    "recursion_limit": ...,
    "configurable": {"thread_id": "..."}
}
```

5. 每次 LangGraph 写 checkpoint 时，`FileCheckpointSaver` 将 MemorySaver 内部状态原子写入本地 pickle 文件

## 5. Resume 流程

1. 调用方设置 `debug.resume: true`
2. 任务入口用同一个 `thread_id` 编译并加载 checkpoint 文件
3. 执行前调用 `app.aget_state(invoke_config)` 检查已保存状态
4. 若不存在 checkpoint，立即报错，不从头静默重跑
5. 若存在 checkpoint，使用 `app.ainvoke(None, config=invoke_config)` 交给 LangGraph 从上次状态继续

该逻辑已覆盖：

- `BaseLangGraphTask.run()`
- `LangGraphTask.run()`
- `ToolExecutor._execute_workflow()`
- `OpTaskBuilderWorkflow.run()`

## 6. 与 Trace / Replay 的关系

Debug checkpoint 保存的是 LangGraph state，用于恢复图执行状态。

Trace / WorkflowLogger 保存的是可读日志和节点产物，用于人工定位问题。

ReplayGuard / LLM cache 用于复现 LLM 输出。开启 checkpoint debug 时，`LangGraphTask` 不会把 `_replay_guard` 对象写入 state，避免不可序列化对象污染 checkpoint。

三者不是替代关系：

- checkpoint 解决“从哪里继续”
- Trace 解决“发生了什么”
- Replay/cache 解决“如何复现同一批 LLM 输出”

## 7. 失败行为

- checkpoint 文件损坏：加载时报 `RuntimeError`，提示具体文件路径
- `debug.resume=true` 但无历史 state：报错终止
- 未开启 debug：保持原始 `graph.compile()` 和 `ainvoke()` 行为
- checkpoint 使用 pickle，仅适用于可信本地调试文件，不应加载外部不可信文件

## 8. 验收

运行定向单测：

```bash
PYTHONPATH=/workspace/akg/akg_agents/python \
pytest -q akg_agents/tests/ut/test_langgraph_debug_checkpoint.py
```

期望结果：

- 首次执行生成 checkpoint 文件
- 新 graph 实例能读取已有 state
- `ainvoke(None, config=...)` 能从 checkpoint resume
