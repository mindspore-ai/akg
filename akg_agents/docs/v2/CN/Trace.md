[English Version](../Trace.md)

# Trace 追踪系统

## 1. 概述

Trace 系统为 AKG Agents 提供树状推理追踪能力，记录 Agent 的完整推理和执行过程，支持：

- **树状结构**：单一树，支持多分叉探索
- **节点切换**：可导航到 trace 树中的任意节点
- **增量历史**：每个节点累积动作记录
- **断点续跑**：持久化状态，从任意点恢复
- **可视化**：文本和 Rich 终端可视化
- **分叉与合并**：在 ask_user 节点分叉、分支合并、并行探索
- **Blame**：追踪生成代码每一行由哪个节点引入

## 2. 核心概念

### Trace 树

整个 trace 是一棵树 —— 没有单独的"分支"概念。一个节点可以有多个子节点（分叉）。当在已有子节点的节点上执行新操作时，自动创建新的子节点。

### 数据模型

| 模型 | 说明 |
|------|------|
| `TraceTree` | 根数据结构，包含所有节点和元数据 |
| `TraceNode` | 树中的单个节点（Agent 信息、任务信息、状态、子节点） |
| `NodeState` | 节点状态快照 |
| `ActionRecord` | 单条动作记录（工具调用 + 结果） |
| `ActionHistoryFact` | 每个节点累积的事实性动作历史 |
| `ActionHistoryCompressed` | 压缩后的动作历史，用于 LLM 上下文 |
| `AgentInfo` | Agent 元数据（名称、ID） |
| `TaskInfo` | 任务元数据（task_id、输入、领域、自定义元数据） |
| `ExecutionInfo` | 执行计数器（工具调用次数、轮次） |
| `FileInfo` | 代码快照的文件元数据 |
| `Metrics` | 节点的性能和质量指标 |
| `ThinkingState` | 规划与决策状态（thinking.json） |
| `PlanState` | ThinkingState 中的结构化规划状态 |
| `DecisionRecord` | 单条思考/决策记录 |
| `PendingTool` | 等待完成的工具调用 |
| `PendingToolsState` | 待处理工具集合（pending_tools.json） |

## 3. TraceSystem

`TraceSystem` 是 trace 管理的主入口。

### 构造函数

```python
trace = TraceSystem(task_id="my_task", base_dir="~/.akg")
```

### 初始化

```python
# 初始化（创建 trace.json 和 root 节点）
trace.initialize(task_input="帮我生成一个 relu 算子")

# 强制重新初始化（覆盖已有 trace）
trace.initialize(task_input="帮我生成一个 relu 算子", force=True)

# 恢复已有 trace（如果 trace.json 存在则自动加载）
trace.initialize(force=False)
```

### 节点操作

```python
# 添加新节点（返回 node_id）
node_id = trace.add_node(
    action={"type": "verify_kernel", "arguments": {"op_name": "relu"}},
    result={"status": "success", "output": "..."},
    metrics={"performance": 1.5},          # 可选
    state_snapshot={"iteration": 3},       # 可选
)

# 获取当前节点 ID
current = trace.get_current_node()

# 获取节点对象
node = trace.get_node(node_id)

# 切换到其他节点
trace.switch_node(node_id)

# 获取从根节点到指定节点的路径
path = trace.get_path_to_node(node_id)

# 获取节点在树中的深度
depth = trace.get_node_depth(node_id)
```

### 动作历史

```python
# 获取节点的完整动作历史（包含所有祖先节点）
history = trace.get_full_action_history(node_id)

# 获取压缩后的 LLM 上下文历史（异步，需要 LLM 客户端）
compressed = await trace.get_compressed_history_for_llm(
    llm_client, node_id, max_tokens=2000
)
```

### 分叉与合并

```python
# 在 ask_user 节点分叉（创建新子节点，保留问题，清空回答）
new_node_id = trace.fork_ask_user(node_id)

# 创建并行探索分叉
fork_ids = trace.create_parallel_forks(n=3, action_template={"type": "explore"})

# 完成分叉并提交结果
trace.complete_fork(fork_id, result={"status": "success"}, metrics={...})

# 合并两个分支（三路合并代码文件）
merged_node_id = trace.merge_nodes(target_node_id, source_node_id)
```

### 节点状态

```python
# 标记节点为已完成
trace.mark_node_completed(node_id, metrics={"performance": 1.5})

# 标记节点为失败
trace.mark_node_failed(node_id, error="验证超时")

# 更新节点结果
trace.update_node_result(node_id, result={...}, metrics={...})
```

### 树查询

```python
# 获取所有叶子节点
leaves = trace.get_all_leaf_nodes()

# 获取按指标排序的最佳叶子节点
best = trace.get_best_leaf_node(metric="performance")

# 对比两个节点
diff = trace.compare_nodes(node_1, node_2)

# 查找最近公共祖先
lca = trace.find_lca(node_id1, node_id2)
```

### 代码 Blame

```python
# Blame：追踪每一行代码由哪个节点引入
blame_info = trace.blame_file(node_id, "kernel.py")

# Blame 节点下的所有文件
all_blame = trace.blame_all_files(node_id)
```

### 恢复

```python
# 获取断点续跑信息
resume_info = trace.get_resume_info()
```

## 4. FileSystemState

`FileSystemState` 负责将 trace 数据持久化到文件系统。

```
~/.akg/tasks/{task_id}/
├── trace.json              # Trace 树结构
├── .traceconfig            # Trace 配置
└── nodes/
    ├── root/
    │   ├── state.json      # 节点状态
    │   ├── result.json     # 动作结果
    │   ├── action_history_fact.json
    │   ├── thinking.json   # 规划/决策状态
    │   ├── pending_tools.json
    │   ├── code/           # 代码快照（CoW）
    │   ├── logs/           # 验证日志等
    │   └── system_prompts/ # 每轮系统 prompt
    ├── node_001/
    │   ├── state.json
    │   ├── result.json
    │   └── ...
    └── ...
```

### 核心方法

| 方法 | 说明 |
|------|------|
| `save_node_state(node_id, state)` | 将节点状态持久化到 `state.json` |
| `load_node_state(node_id)` | 从 `state.json` 加载节点状态 |
| `append_action(node_id, action)` | 向节点的历史中追加动作记录 |
| `save_code_file(node_id, filename, content)` | 保存代码文件快照（CoW） |
| `load_code_file(node_id, filename)` | 从节点快照加载代码文件 |
| `diff_nodes(node_a, node_b)` | 生成两个节点代码的 diff |
| `copy_node_state(from_node, to_node)` | 在节点间复制状态 |
| `save_system_prompt(node_id, turn, prompt)` | 保存每轮系统 prompt |

## 5. ActionCompressor

`ActionCompressor` 压缩动作历史以适应 LLM 上下文窗口。它会摘要较旧的动作，同时保留最近的动作完整细节。

```python
from akg_agents.core_v2.filesystem import ActionCompressor

compressor = ActionCompressor(llm_client)
compressed = await compressor.compress_history(action_history, max_tokens=4000)
```

> 注意：`ActionCompressor` 需要一个 `LLMClient` 实例用于摘要生成。

## 6. 可视化

### 文本可视化

```python
from akg_agents.core_v2.filesystem.trace_visualizer import visualize_text

text = visualize_text(trace, focus_node="node_005", depth=4)
print(text)
```

### Rich 终端可视化

```python
from akg_agents.core_v2.filesystem.trace_visualizer import visualize_rich

rich_text = visualize_rich(trace, focus_node="node_005", depth=4)
```

### 节点详情

```python
from akg_agents.core_v2.filesystem.trace_visualizer import format_node_detail_rich

detail = format_node_detail_rich(trace, "node_003")
```

## 7. CLI：/trace 命令

在 `akg_cli` 交互模式下，`/trace` 斜杠命令（别名：`/t`）提供 trace 树查看和分叉功能。

### 用法

```
/trace [<id>|root|show <id>|node <id>|history|fork <id>] [-n depth]
```

### 子命令

| 子命令 | 示例 | 说明 |
|--------|------|------|
| （无参数） | `/trace` | 以当前节点为中心显示路径视图（默认深度 4） |
| `root` | `/trace root` | 从根节点向下显示 trace 树 |
| `<id>` | `/trace 005` | 以指定节点为中心显示路径视图（`005` 自动补全为 `node_005`） |
| `-n <N>` | `/trace -n 8` | 设置显示深度为 N |
| `show <id>` | `/trace show 003` | 显示节点详细信息（动作、参数、结果） |
| `node <id>` | `/trace node 003` | 同 `show` |
| `history` | `/trace history` | 显示当前节点的完整动作历史 |
| `fork <id>` | `/trace fork 005` | 在 `ask_user` 节点创建分叉，给出不同回答 |

### 分叉行为

`/trace fork <id>` 仅对 `ask_user` 类型节点有效。执行后：

1. 复制原 ask_user 节点的动作（问题），清除用户回答
2. 在同一父节点下创建新子节点
3. 更新 agent 的当前节点和动作历史
4. 重新展示原问题，等待用户输入新回答

### 示例

```bash
# 查看 trace 树
/trace

# 从根节点查看，深度 8
/trace root -n 8

# 查看节点 003 的详情
/trace show 003

# 在节点 005 分叉，尝试不同回答
/trace fork 005
```

## 8. 异常

| 异常 | 说明 |
|------|------|
| `FileSystemStateError` | 文件系统状态错误基类 |
| `NodeNotFoundError` | 节点在 trace 中不存在 |
| `TraceSystemError` | 通用 trace 系统错误 |
| `InvalidNodeStateError` | 节点状态无效 |
| `TraceNotInitializedError` | Trace 未初始化 |
| `TraceAlreadyExistsError` | Trace 已存在（force=False 时） |
