# TraceSystem 设计文档

## 概述
TraceSystem 是 AKG Agents 中基于树结构的推理追踪系统，负责完整记录 AI Kernel 生成过程中的所有操作轨迹。它支持多分叉探索、节点切换、增量动作历史和断点续跑，基于 `FileSystemState` 实现持久化存储。

## 架构概览

```
core_v2/filesystem/
├── trace_system.py         # TraceSystem 主类
├── state.py                # FileSystemState 文件系统状态管理
├── models.py               # 数据模型定义
├── compressor.py           # ActionCompressor 动作压缩
└── exceptions.py           # 自定义异常

存储目录结构：
~/.akg_agents/conversations/{task_id}/
├── trace.json              # Trace 树结构
├── current_node.txt        # 当前活动节点
├── nodes/
│   ├── root/
│   │   └── state.json
│   ├── node_001/
│   │   ├── state.json                  # 节点状态快照
│   │   ├── thinking.json               # 思考过程
│   │   ├── actions/
│   │   │   ├── action_history_fact.json      # 增量动作历史
│   │   │   ├── action_history_compressed.json # 压缩动作历史
│   │   │   └── pending_tools.json            # 待执行工具
│   │   ├── code/                       # 代码文件
│   │   └── system_prompts/             # 系统提示词
│   └── ...
└── logs/
```

## 核心概念

### Trace 树（TraceTree）
整个追踪记录就是一棵树，没有"分支"的概念。每个节点可以有多个子节点（children），形成分叉。存储在 `trace.json` 中。

### 节点（TraceNode）
树中的每个节点记录：
- `node_id`：唯一标识（如 `root`、`node_001`）
- `parent_id`：父节点
- `action`：执行的动作
- `result`：执行结果
- `metrics`：指标数据（token 使用、耗时、性能等）
- `state_snapshot`：状态快照（turn、status）
- `children`：子节点列表

### 动作记录（ActionRecord）
单个工具调用的记录：
- `action_id`：动作唯一标识
- `tool_name`：工具名称
- `arguments`：调用参数
- `result`：执行结果
- `duration_ms`：执行耗时

### 增量动作历史（ActionHistoryFact）
每个节点只保存**自己新增**的动作（增量），避免重复存储。完整历史通过沿 parent 链回溯重建。

### 压缩动作历史（ActionHistoryCompressed）
通过 LLM 压缩的历史摘要，用于减少 token 消耗。带缓存机制，通过 `source_path` 校验缓存有效性。

## 核心 API

### 初始化
```python
from akg_agents.core_v2.filesystem.trace_system import TraceSystem

# 创建 TraceSystem
trace = TraceSystem(task_id="my_task", base_dir="~/.akg_agents")

# 初始化（创建 trace.json 和 root 节点）
trace.initialize(task_input="用户请求内容")

# 断点续跑（自动加载已有 trace）
trace.initialize()  # 如果 trace 已存在则加载
```

### 节点操作
```python
# 获取当前节点
current = trace.get_current_node()  # → "root" 或 "node_001"

# 添加子节点（自动在当前节点下创建）
node_id = trace.add_node(
    action={"type": "call_kernel_gen", "arguments": {...}},
    result={"status": "success", "code": "..."},
    metrics={"token_used": 1500, "duration_ms": 3200}
)

# 更新节点结果
trace.update_node_result(
    node_id="node_001",
    result={"status": "success", "performance": 0.95},
    metrics={"token_used": 500}
)

# 标记节点完成/失败
trace.mark_node_completed("node_001", metrics={"performance": 0.95})
trace.mark_node_failed("node_001", error="验证失败：精度不达标")

# 切换到指定节点
trace.switch_node("node_001")
```

### 路径与历史
```python
# 获取从 root 到节点的路径
path = trace.get_path_to_node("node_003")  # → ["root", "node_001", "node_002", "node_003"]

# 获取完整动作历史（沿 parent 链回溯重建）
history = trace.get_full_action_history("node_003")

# 获取 LLM 用的压缩历史（带缓存）
compressed = await trace.get_compressed_history_for_llm(
    llm_client=client,
    node_id="node_003",
    max_tokens=2000
)
```

### 并行分叉
```python
# 创建 N 个并行分叉节点
fork_nodes = trace.create_parallel_forks(
    n=3,
    action_template={"type": "parallel_gen"}
)
# → ["node_004", "node_005", "node_006"]

# 完成分叉节点
trace.complete_fork(
    node_id="node_004",
    result={"status": "success", "code": "..."},
    metrics={"performance": 0.92}
)
```

### 节点对比
```python
# 对比两个节点的路径
comparison = trace.compare_nodes("node_003", "node_005")
# → {
#     "path_1": [...], "path_2": [...],
#     "fork_point": "node_001",
#     "metrics_1": {"steps": 3, "total_token": 4500, ...},
#     "metrics_2": {"steps": 2, "total_token": 3000, ...}
# }

# 获取最优叶节点
best = trace.get_best_leaf_node(metric="performance")
```

### 可视化
```python
# 打印树的字符串表示
print(trace.visualize_tree())
# 输出示例：
# 🌳 Trace Tree (当前: node_003):
# root
#   ↓
#   [001] call_kernel_gen
#       → ✅ | 150 行
#     ↓
#     ├─ [002] call_verifier
#     │      → ❌
#     │    ↓
#     │    [003] call_kernel_gen ⭐ 当前
#     └─ [004] call_kernel_gen
#            → ✅ | 性能: 92.0%

# 获取节点详情
print(trace.get_node_detail("node_001"))

# 获取路径详情
print(trace.get_path_detail("node_003"))
```

### 任务恢复
```python
# 获取恢复信息（用于断点续跑）
resume_info = trace.get_resume_info()
# → {
#     "task_id": "my_task",
#     "current_node": "node_003",
#     "state": NodeState(...),
#     "action_history": [...],
#     "pending_tools": PendingToolsState(...),
#     "thinking": ThinkingState(...),
#     "path": ["root", "node_001", "node_003"]
# }
```

## 数据模型

### NodeState（节点状态快照）
对应 `nodes/{node_id}/state.json`：

| 字段 | 类型 | 说明 |
|------|------|------|
| node_id | str | 节点 ID |
| turn | int | 轮次 |
| status | str | 状态：`init` / `running` / `completed` / `failed` |
| agent_info | dict | Agent 信息 |
| task_info | dict | 任务信息 |
| execution_info | dict | 执行信息 |
| file_state | dict | 文件状态（path → FileInfo） |
| metrics | dict | 指标数据 |

### ThinkingState（思考过程）
对应 `nodes/{node_id}/thinking.json`：

| 字段 | 类型 | 说明 |
|------|------|------|
| node_id | str | 节点 ID |
| turn | int | 轮次 |
| current_plan | dict | 当前计划 |
| latest_thinking | str | 最新思考 |
| decision_history | list | 决策历史 |

### PendingToolsState（待执行工具）
对应 `nodes/{node_id}/actions/pending_tools.json`，用于断点续跑：

| 字段 | 类型 | 说明 |
|------|------|------|
| node_id | str | 节点 ID |
| turn | int | 轮次 |
| pending_tools | list | 待执行工具列表 |

## FileSystemState

`FileSystemState` 是底层文件系统状态管理器，所有持久化操作通过它完成。核心原则：
1. **文件系统作为真相源**：所有状态都持久化到文件系统
2. **人类可读**：使用 JSON/Text 格式，便于调试和审计
3. **增量保存**：避免重复，节省空间
4. **框架无关**：不依赖特定框架，易于迁移
5. **支持断点续跑**：保存足够信息以恢复执行

## 相关文档
- [Workflow 与任务系统](./Workflow.md)
- [Skill System 文档](./SkillSystem.md)
