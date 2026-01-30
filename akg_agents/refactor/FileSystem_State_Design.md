# AIKG FileSystem 状态管理设计

> 基于文件系统的状态持久化方案

---

## 一、设计理念

### 核心原则

1. **文件系统作为真相源**：所有状态都持久化到文件系统
2. **人类可读**：使用 JSON/Text 格式，便于调试和审计
3. **增量保存**：避免重复，节省空间
4. **框架无关**：不依赖特定框架，易于迁移
5. **支持断点续跑**：保存足够信息以恢复执行

---

## 二、目录结构

```
~/.akg/conversations/{task_id}/
├─ trace.json                      # Trace 树结构 ⭐
├─ current_node.txt                # 当前节点 ID
├─ checkpointer.db                 # LangGraph Checkpointer (SQLite)
│
├─ nodes/                          # 每个节点的状态 ⭐
│   ├─ root/
│   │   └─ state.json
│   │
│   ├─ node_001/
│   │   ├─ state.json              # 节点状态快照
│   │   ├─ thinking.json           # PlanAgent 固化
│   │   ├─ actions/                # 动作历史
│   │   │   ├─ action_history.json          # 压缩版（渲染用）
│   │   │   ├─ action_history_fact.json     # 增量版（审计用）⭐
│   │   │   └─ pending_tools.json           # 待执行工具 ⭐
│   │   ├─ code/                   # 生成的代码
│   │   │   ├─ kernel.cu
│   │   │   └─ ...
│   │   └─ system_prompts/         # 可选：调试用
│   │       ├─ turn_001.txt
│   │       └─ turn_002.txt
│   │
│   ├─ node_002/
│   └─ ...
│
└─ logs/
```

---

## 三、核心文件详解

### 3.1 `trace.json` - Trace 树结构

**用途**：管理整个任务的节点关系和分叉

```json
{
  "task_id": "task_20260121_001",
  "created_at": "2026-01-21T10:00:00Z",
  "current_node": "node_003",
  
  "tree": {
    "root": {
      "node_id": "root",
      "parent_id": null,
      "state_snapshot": {
        "turn": 0,
        "status": "init"
      },
      "action": null,
      "result": null,
      "timestamp": "2026-01-21T10:00:00Z",
      "children": ["node_001"],
      "metrics": {},
      "checkpointer_thread": null
    },
    
    "node_001": {
      "node_id": "node_001",
      "parent_id": "root",
      "state_snapshot": {
        "turn": 1,
        "status": "running"
      },
      "action": {
        "type": "call_designer",
        "params": {}
      },
      "result": {
        "success": true,
        "output": "设计方案..."
      },
      "timestamp": "2026-01-21T10:05:00Z",
      "children": ["node_002"],
      "metrics": {
        "token_used": 1200,
        "duration_ms": 5000
      },
      "checkpointer_thread": "thread_001"
    },
    
    "node_002": {
      "node_id": "node_002",
      "parent_id": "node_001",
      "children": ["node_003", "node_004"],
      "checkpointer_thread": "thread_002",
      "metrics": {...}
    }
  }
}
```

**关键字段**：
- `tree`: 节点关系（parent/children）
- `checkpointer_thread`: 链接到 LangGraph Checkpointer 的 thread_id

---

### 3.2 `nodes/{node_id}/state.json` - 节点状态快照

**用途**：保存节点执行时的完整状态

```json
{
  "node_id": "node_001",
  "turn": 1,
  "status": "running",
  
  "agent_info": {
    "agent_name": "KernelAgent",
    "agent_id": "KernelAgent_abc123"
  },
  
  "task_info": {
    "task_input": "实现一个 softmax kernel",
    "task_id": "task_20260121_001"
  },
  
  "execution_info": {
    "tool_call_counter": 5,
    "first_thinking_done": true,
    "current_turn": 1
  },
  
  "file_state": {
    "code/kernel.cu": {
      "size": 1024,
      "last_modified": "2026-01-21T10:05:00Z",
      "checksum": "abc123..."
    }
  },
  
  "metrics": {
    "token_used": 1200,
    "duration_ms": 5000,
    "performance": null
  },
  
  "timestamp": "2026-01-21T10:05:00Z"
}
```

---

### 3.3 `nodes/{node_id}/thinking.json` - PlanAgent 固化

**用途**：保存 PlanAgent 的规划信息（每10步固化一次）

```json
{
  "node_id": "node_001",
  "turn": 1,
  
  "current_plan": {
    "goal": "实现一个高性能的 softmax kernel",
    "steps": [
      "分析需求，确定算法策略",
      "设计数据布局和 memory access pattern",
      "实现代码",
      "验证正确性",
      "优化性能"
    ],
    "current_step": 2,
    "status": "in_progress"
  },
  
  "latest_thinking": "当前我需要优化 memory access pattern，减少 bank conflict...",
  
  "decision_history": [
    {
      "turn": 1,
      "thinking": "首先需要设计算法策略...",
      "decision": "使用 Block Reduce + Warp Shuffle",
      "timestamp": "2026-01-21T10:05:00Z"
    }
  ],
  
  "timestamp": "2026-01-21T10:05:00Z"
}
```

---

### 3.4 `nodes/{node_id}/actions/action_history.json` - 压缩版动作历史

**用途**：用于渲染、展示（减少 token 消耗）

```json
{
  "node_id": "node_001",
  "turn": 1,
  
  "actions": [
    {
      "action_id": "action_001",
      "tool_name": "file_read",
      "arguments": {"path": "kernel.cu"},
      "result": {
        "status": "success",
        "content": "...(truncated)"
      },
      "timestamp": "2026-01-21T10:05:00Z",
      "compressed": false
    },
    {
      "action_id": "action_002",
      "tool_name": "call_coder",
      "arguments": {"strategy": "optimize"},
      "result": {
        "status": "success",
        "summary": "生成优化后的代码"
      },
      "timestamp": "2026-01-21T10:10:00Z",
      "compressed": true
    }
  ],
  
  "total_actions": 15,
  "compressed_count": 10,
  "last_updated": "2026-01-21T10:15:00Z"
}
```

---

### 3.5 `nodes/{node_id}/actions/action_history_fact.json` - 增量版动作历史 ⭐

**用途**：完整保存，用于审计、恢复、调试

**关键设计**：每个 node 只保存自己新增的 action，避免重复

```json
{
  "node_id": "node_003",
  "parent_node_id": "node_002",
  "turn": 3,
  
  "actions": [
    {
      "action_id": "action_005",
      "tool_name": "verify_tool",
      "arguments": {"path": "kernel.cu"},
      "result": {
        "status": "success",
        "performance": 65,
        "details": "...（完整内容，不截断）"
      },
      "timestamp": "2026-01-21T10:15:00Z",
      "duration_ms": 1000
    }
    // 只保存在 node_003 执行的 action
  ],
  
  "actions_count": 1,
  "last_updated": "2026-01-21T10:15:00Z"
}
```

**为什么是增量保存？**

假设路径：`root → node_001 → node_002 → node_003`

```
❌ 完整保存（有重复）：
node_001: [action_001, action_002]  // 2 个
node_002: [action_001, action_002, action_003, action_004]  // 4 个（重复）
node_003: [action_001, action_002, action_003, action_004, action_005]  // 5 个（重复）
总存储: 11 个 action（实际只有 5 个）

✅ 增量保存（无重复）：
node_001: [action_001, action_002]  // 只保存 node_001 的
node_002: [action_003, action_004]  // 只保存 node_002 的
node_003: [action_005]  // 只保存 node_003 的
总存储: 5 个 action（正好）
```

**如何重建完整历史？**

```python
def get_full_action_history(node_id: str) -> List[Dict]:
    """沿 parent 链回溯，重建完整 action history"""
    # 1. 获取路径: ["root", "node_001", "node_002", "node_003"]
    path = get_path_to_node(node_id)
    
    # 2. 收集所有 node 的 action
    full_history = []
    for node in path:
        if node != "root":
            actions = load_json(f"nodes/{node}/actions/action_history_fact.json")
            full_history.extend(actions["actions"])
    
    return full_history
```

---

### 3.6 `nodes/{node_id}/actions/pending_tools.json` - 待执行工具 ⭐

**用途**：支持断点续跑

```json
{
  "node_id": "node_001",
  "turn": 1,
  
  "pending_tools": [
    {
      "tool_call_id": "call_abc123",
      "tool_name": "file_write",
      "arguments": {
        "path": "output.cu",
        "content": "..."
      },
      "status": "pending",
      "created_at": "2026-01-21T10:15:00Z"
    }
  ],
  
  "last_updated": "2026-01-21T10:15:00Z"
}
```

---

### 3.7 `nodes/{node_id}/system_prompts/turn_XXX.txt` - 完整 System Prompt（可选）⭐

**用途**：调试时查看 LLM 看到的完整上下文

```
# Task
实现一个 softmax kernel

## Requirements
- 高性能
- 支持 batch processing

# Current Plan
Step 1: 分析需求 ✅
Step 2: 设计数据布局 ← 当前
Step 3: 实现代码
...

# Available Tools
1. file_read: 读取文件
2. file_write: 写入文件
3. call_coder: 调用 Coder SubAgent
...

# Action History (compressed)
[action_001] file_read(kernel.cu) → success
[action_002] call_designer(...) → success
...

# File State
- code/kernel.cu (1024 bytes)
- code/test.py (512 bytes)
...

# Thinking
当前需要优化 memory access pattern...
```

**配置**：
- 开发阶段：开启（`save_system_prompts=True`）
- 生产环境：关闭（节省空间）

---

### 3.8 `checkpointer.db` - LangGraph Checkpointer

**用途**：LangGraph 管理 Workflow/SubAgent 内部状态

**保存内容**：
- LangChain Message 对象（序列化）
- Graph State
- Thread ID

**与 FileSystem State 的关系**：
- **Checkpointer**: 管理 Workflow 内部状态（fine-grained）
- **FileSystem State**: 管理 Node 级别状态、Trace 树（coarse-grained）

**使用场景**：
- Workflow/SubAgent 使用 LangGraph + Checkpointer
- KernelAgent 使用 FileSystem State（不用 LangGraph）

---

## 四、状态恢复流程

### 完整恢复流程

```python
def resume_task(task_id: str):
    """恢复任务执行"""
    
    # 1. 读取 Trace Tree
    trace = json.load(open(f"~/.akg/conversations/{task_id}/trace.json"))
    current_node = trace["current_node"]
    
    # 2. 读取节点状态
    state = json.load(open(f"~/.akg/conversations/{task_id}/nodes/{current_node}/state.json"))
    turn = state["turn"]
    tool_call_counter = state["execution_info"]["tool_call_counter"]
    
    # 3. 重建完整 action_history_fact（沿 parent 链回溯）
    action_history_fact = get_full_action_history(current_node)
    
    # 4. 读取 pending_tools
    pending_tools = json.load(
        open(f"~/.akg/conversations/{task_id}/nodes/{current_node}/actions/pending_tools.json")
    )
    
    # 5. 读取 thinking
    thinking = json.load(
        open(f"~/.akg/conversations/{task_id}/nodes/{current_node}/thinking.json")
    )
    
    # 6. 读取 system_prompt（可选）
    system_prompt = None
    prompt_file = f"~/.akg/conversations/{task_id}/nodes/{current_node}/system_prompts/turn_{turn:03d}.txt"
    if Path(prompt_file).exists():
        system_prompt = open(prompt_file).read()
    
    # 7. 恢复 LangGraph Checkpointer（如果该 node 使用了 Workflow）
    checkpointer = None
    thread_id = trace["tree"][current_node].get("checkpointer_thread")
    if thread_id:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(
            f"~/.akg/conversations/{task_id}/checkpointer.db"
        )
    
    # 8. 继续执行
    kernel_agent.resume(
        state=state,
        action_history=action_history_fact,
        pending_tools=pending_tools,
        thinking=thinking,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        thread_id=thread_id
    )

def get_full_action_history(node_id: str) -> List[Dict]:
    """获取从 root 到指定 node 的完整 action history"""
    trace = json.load(open("trace.json"))
    
    # 1. 获取路径
    path = []
    current = node_id
    while current != "root":
        path.append(current)
        current = trace["tree"][current]["parent_id"]
    path.reverse()
    
    # 2. 收集所有 action
    full_history = []
    for node in path:
        actions_file = f"nodes/{node}/actions/action_history_fact.json"
        if Path(actions_file).exists():
            actions_data = json.load(open(actions_file))
            full_history.extend(actions_data["actions"])
    
    return full_history
```

---

## 五、调试流程

### 查看某个 node 的状态

```bash
# 1. 查看 state
$ cat ~/.akg/conversations/task_001/nodes/node_003/state.json | jq .

# 2. 查看完整 action history
$ cat ~/.akg/conversations/task_001/nodes/node_003/actions/action_history_fact.json | jq .

# 3. 查看 LLM 看到的 prompt
$ cat ~/.akg/conversations/task_001/nodes/node_003/system_prompts/turn_005.txt

# 4. 查看 thinking
$ cat ~/.akg/conversations/task_001/nodes/node_003/thinking.json | jq .
```

### 对比两个 node 的差异

```bash
# 对比 prompt
$ diff \
    ~/.akg/conversations/task_001/nodes/node_003/system_prompts/turn_005.txt \
    ~/.akg/conversations/task_001/nodes/node_004/system_prompts/turn_005.txt

# 对比 state
$ diff <(cat ~/.akg/conversations/task_001/nodes/node_003/state.json | jq -S .) \
       <(cat ~/.akg/conversations/task_001/nodes/node_004/state.json | jq -S .)
```

---

## 六、设计优势

### 1. 人类可读 ✅

- 所有文件都是 JSON/Text 格式
- 可以直接用 `cat`, `jq`, `grep` 查看
- 易于调试和审计

### 2. 增量保存，节省空间 ✅

- 每个 node 只保存自己的 action
- 多分支场景节省 ~30-40% 空间
- 需要完整历史时，沿 parent 链回溯

### 3. 支持断点续跑 ✅

- `pending_tools.json` 保存未执行的工具
- 完整保存所有状态信息
- 可以从任意 node 恢复

### 4. 框架无关 ✅

- 不依赖 LangGraph（只在 Workflow 中使用）
- 可以迁移到其他框架
- FileSystem State 是通用方案

### 5. 灵活的 Trace Tree ✅

- 支持多分支探索
- 可以切换到任意 node
- 可以对比不同路径

---

## 七、实现建议

### P0（必须）

- `trace.json` 基础结构
- `nodes/{node_id}/state.json`
- `nodes/{node_id}/actions/action_history_fact.json`（增量保存）
- `TraceSystem.add_node()` - 添加节点
- `TraceSystem.get_full_action_history()` - 重建完整历史

### P1（重要）

- `nodes/{node_id}/thinking.json`
- `nodes/{node_id}/actions/pending_tools.json`
- `TraceSystem.switch_node()` - 切换节点
- 恢复流程实现

### P2（增强）

- `nodes/{node_id}/system_prompts/` - 保存完整 prompt（可选）
- `checkpointer.db` - LangGraph 集成
- 对比功能

---

**总结**：
- FileSystem 作为真相源，所有状态持久化
- 增量保存避免重复，节省空间
- 人类可读，易于调试
- 支持断点续跑和多分支探索
- 框架无关，易于迁移
