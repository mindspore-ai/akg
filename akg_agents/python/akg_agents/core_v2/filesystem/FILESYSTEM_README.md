# AIKG FileSystem API 文档

## 概述

`core_v2.filesystem` 模块提供基于文件系统的状态持久化方案，支持：
- 树状 Trace 管理（分叉、合并、回溯）
- 节点状态快照
- 增量动作历史保存
- LLM 历史压缩与缓存
- 断点续跑

---
 
## 快速开始

```python
from akg_agents.core_v2.filesystem import (
    FileSystemState,
    TraceSystem,
    ActionRecord,
    NodeState,
    ThinkingState,
    PendingTool,
    PendingToolsState,
)

# 初始化
task_id = "my_task_001"
fs = FileSystemState(task_id, base_dir="/path/to/storage")
trace = TraceSystem(task_id, base_dir="/path/to/storage")

# 初始化任务
fs.initialize_task()
trace.initialize()

# 记录动作
node_id = trace.add_node(
    action={"type": "coder", "params": {"iteration": 1}},
    result={"success": True, "code_len": 500},
)

# 保存代码
fs.save_code_file(node_id, "kernel.py", generated_code)

# 获取完整历史
history = trace.get_full_action_history(node_id)

# 获取压缩历史（用于 LLM 上下文）
compressed_actions = await trace.get_compressed_history_for_llm(llm_client, node_id)
```

---

## 模块结构

```
core_v2/filesystem/
├── __init__.py          # 导出所有公开 API
├── models.py            # 数据模型
├── exceptions.py        # 异常类
├── state.py             # FileSystemState 类
├── trace_system.py      # TraceSystem 类
├── compressor.py        # ActionCompressor 类 
└── FILESYSTEM_README.md # 本文档 (原 API.md)
```

---

## 深入了解：节点与存储结构

### 1. 节点命名规则

AIKG 任务中的节点 ID 生成规则旨在保持人类可读性和唯一性：

*   **根节点 (Root Node)**: 固定命名为 `"root"`。它是 Trace 树的起点。
*   **普通节点**: 采用自增格式命名，如 `node_001`, `node_002`, `node_003` 等。
    *   系统内部维护一个全局计数器 (`_node_counter`)。
    *   每次添加新节点时，计数器加 1。
    *   这种设计方便在文件系统中按顺序查看节点目录，同时也便于调试。

### 2. 节点目录详细结构

每个节点在文件系统中拥有一个独立的目录（例如 `nodes/node_001/`），其中包含该节点的所有状态和数据。

```text
{base_dir}/conversations/{task_id}/nodes/node_001/
├── state.json                  # [状态快照] 包含节点状态、执行结果、Metrics 等
├── thinking.json               # [思考过程] 包含 Agent 的思维链、Plan 和决策
├── actions/                    # [动作历史]
│   ├── action_history_fact.json        # 增量完整历史 (Source of Truth)
│   ├── action_history_compressed.json  # 压缩历史缓存 (Derived Cache)
│   └── pending_tools.json              # 待处理工具 (HITL)
├── code/                       # [代码文件]
│   └── kernel.py               # 该节点生成或修改的代码快照
└── system_prompts/             # [系统提示词]
    └── turn_001.txt            # 该轮次使用的 System Prompt 存档
```

#### 关键文件说明：
- **`state.json`**: 记录节点的核心 metadata，如 status ("completed"/"failed")，turn (轮次)，execution_info 等。
- **`action_history_fact.json`**: 仅记录在该节点**新产生**的动作（增量）。通过 TraceSystem 回溯父节点链并合并所有 path 上的 fact，即可重建完整历史。
- **`action_history_compressed.json`**: LLM 压缩后的历史摘要。包含 `source_path` 字段用于校验缓存是否过期（例如父节点链发生变化）。

---

## 类参考

### 1. FileSystemState

文件系统状态管理器，负责节点级别的持久化。

#### 构造函数

```python
FileSystemState(task_id: str, base_dir: str = None)
```

| 参数       | 类型 | 说明                       |
| ---------- | ---- | -------------------------- |
| `task_id`  | str  | 任务唯一标识               |
| `base_dir` | str  | 存储根目录，默认 `~/.akg_agents` |

#### 目录结构

```
{base_dir}/conversations/{task_id}/
├── current_node.txt           # 当前节点 ID
├── trace.json                 # Trace 树结构（由 TraceSystem 管理）
├── logs/                      # 日志目录
└── nodes/
    ├── root/
    │   ├── state.json         # 节点状态
    │   ├── thinking.json      # Thinking 状态
    │   ├── actions/
    │   │   ├── action_history_fact.json
    │   │   └── pending_tools.json
    │   ├── code/              # 代码文件
    │   │   └── kernel_v1.py
    │   └── system_prompts/    # 系统提示词
    │       └── turn_001.txt
    └── node_001/
        └── ...
```

#### 核心方法

##### 初始化

```python
def initialize_task(self, force: bool = False) -> None
```
初始化任务目录结构，创建 root 节点。

| 参数    | 说明                               |
| ------- | ---------------------------------- |
| `force` | 是否强制重新初始化（删除现有目录） |

```python
def task_exists(self) -> bool
```
检查任务是否已存在。

##### 当前节点管理

```python
def get_current_node(self) -> str
def set_current_node(self, node_id: str) -> None
```

##### 节点状态管理

```python
def save_node_state(self, node_id: str, state: NodeState) -> None
def load_node_state(self, node_id: str) -> NodeState
def update_node_state(
    self,
    node_id: str,
    turn: int = None,
    status: str = None,
    metrics: Dict = None,
    execution_info: Dict = None,
    file_state: Dict = None,
) -> NodeState
```

##### 动作历史管理

```python
# 增量历史（每个节点只保存自己的动作）
def save_action_history_fact(self, node_id: str, history: ActionHistoryFact) -> None
def load_action_history_fact(self, node_id: str) -> ActionHistoryFact
def append_action(
    self,
    node_id: str,
    action: ActionRecord,
    parent_node_id: str = None,
    turn: int = 0,
) -> ActionHistoryFact

# 压缩历史（用于 LLM 上下文与展示）
def save_action_history_compressed(self, node_id: str, history: ActionHistoryCompressed) -> None
def load_action_history_compressed(self, node_id: str) -> ActionHistoryCompressed
```

##### Thinking 管理

```python
def save_thinking(self, node_id: str, thinking: ThinkingState) -> None
def load_thinking(self, node_id: str) -> Optional[ThinkingState]
def update_thinking(
    self,
    node_id: str,
    latest_thinking: str = None,
    current_plan: Dict = None,
    decision: str = None,
) -> ThinkingState
```

##### Pending Tools 管理（HITL 场景）

```python
def save_pending_tools(self, node_id: str, pending: PendingToolsState) -> None
def load_pending_tools(self, node_id: str) -> PendingToolsState
def add_pending_tool(self, node_id: str, tool: PendingTool, turn: int = 0) -> PendingToolsState
def mark_tool_completed(
    self,
    node_id: str,
    tool_call_id: str,
    raise_if_not_found: bool = False,
) -> PendingToolsState
def clear_pending_tools(self, node_id: str) -> None
```

##### System Prompt 管理

```python
def save_system_prompt(self, node_id: str, turn: int, prompt: str) -> Path
def load_system_prompt(self, node_id: str, turn: int) -> Optional[str]
def get_latest_system_prompt(self, node_id: str) -> Optional[str]
```

##### 代码文件管理

```python
def save_code_file(self, node_id: str, filename: str, content: str) -> Path
def load_code_file(self, node_id: str, filename: str) -> Optional[str]
def list_code_files(self, node_id: str) -> List[str]
def copy_code_files(self, from_node_id: str, to_node_id: str) -> None
def copy_node_state(self, from_node_id: str, to_node_id: str) -> NodeState
```

##### 节点检查与清理

```python
def node_exists(self, node_id: str) -> bool
```
检查节点是否存在。

```python
def delete_node(self, node_id: str) -> None
```
删除节点及其所有数据。**注意**：不能删除 root 节点。

```python
def delete_task(self) -> None
```
删除整个任务及其所有数据。

---

### 2. TraceSystem

树状 Trace 管理器，负责多版本执行路径管理。

#### 构造函数

```python
TraceSystem(task_id: str, base_dir: str = None)
```

#### 核心方法

##### 初始化

```python
def initialize(self, force: bool = False) -> None
```
初始化 Trace 系统。如果已存在会自动加载（断点恢复）。

##### 节点操作

```python
def add_node(
    self,
    action: Dict[str, Any],
    result: Dict[str, Any],
    metrics: Dict[str, Any] = None,
    state_snapshot: Dict[str, Any] = None,
) -> str
```
添加新节点（从当前节点扩展）。如果当前节点已有子节点，自动创建新分支（分叉）；否则直接添加。返回新节点 ID。

| 参数             | 类型           | 说明                         |
| ---------------- | -------------- | ---------------------------- |
| `action`         | Dict[str, Any] | 执行的动作                   |
| `result`         | Dict[str, Any] | 执行结果                     |
| `metrics`        | Dict[str, Any] | 指标数据（可选）             |
| `state_snapshot` | Dict[str, Any] | 状态快照（可选，见下方说明） |

**`state_snapshot` 参数说明**：
- **用途**：保存节点创建时的上下文状态，用于后续回溯和恢复。
- **默认值**：如果不传，系统会自动生成：
  ```json
  {"turn": <父节点turn+1>, "status": "running"}
  ```
- **自定义示例**：可传入更丰富的上下文信息：
  ```python
  state_snapshot={"turn": 5, "iteration": 3, "config": {"max_tokens": 4096}}
  ```


```python
def switch_node(self, node_id: str) -> None
```
切换到指定节点。

```python
def get_node(self, node_id: str) -> TraceNode
```
获取节点信息。返回 `TraceNode` 对象。

| 参数      | 类型 | 说明    |
| --------- | ---- | ------- |
| `node_id` | str  | 节点 ID |

| 返回值   | 类型      | 说明                                                                      |
| -------- | --------- | ------------------------------------------------------------------------- |
| 节点对象 | TraceNode | 包含 `node_id`, `parent_id`, `action`, `result`, `children`, `metrics` 等 |

| 异常                | 说明       |
| ------------------- | ---------- |
| `NodeNotFoundError` | 节点不存在 |

```python
def get_current_node(self) -> str
```
获取当前节点 ID。

##### 节点状态更新

```python
def update_node_result(
    self,
    node_id: str,
    result: Dict[str, Any],
    metrics: Dict[str, Any] = None,
) -> None

def mark_node_completed(self, node_id: str, metrics: Dict = None) -> None
def mark_node_failed(self, node_id: str, error: str = None) -> None
```

##### 路径查询

```python
def get_path_to_node(self, node_id: str) -> List[str]
```
获取从 root 到指定节点的路径。返回 `["root", "node_001", "node_002", ...]`

```python
def get_full_action_history(self, node_id: str) -> List[ActionRecord]
```
获取从 root 到指定节点的完整动作历史（沿 parent 链回溯重建）。

```python
def get_node_depth(self, node_id: str) -> int
```

##### 历史压缩

```python
async def get_compressed_history_for_llm(
    self,
    llm_client,
    node_id: str,
    max_tokens: int = 2000,
    force_refresh: bool = False
) -> List[ActionRecord]
```
获取用于 LLM 上下文渲染的压缩历史。
- 自动检查缓存 (`source_path` 校验)
- 如缓存失效，自动回溯完整历史并调用 `ActionCompressor`
- 返回压缩后的动作列表

##### 节点对比

```python
def compare_nodes(self, node_1: str, node_2: str) -> Dict[str, Any]
```
对比两个节点的路径，返回：
- `path_1`, `path_2`: 两条路径
- `fork_point`: 分叉点
- `metrics_1`, `metrics_2`: 路径累计指标

##### 叶节点查询

```python
def get_all_leaf_nodes(self) -> List[str]
def get_best_leaf_node(self, metric: str = "performance") -> Optional[str]
```

##### 并行分叉（TreeSearch 场景）

```python
def create_parallel_forks(
    self,
    n: int,
    action_template: Dict[str, Any],
) -> List[str]
```
从当前节点创建 N 个并行分叉。返回新节点 ID 列表。

```python
def complete_fork(
    self,
    node_id: str,
    result: Dict[str, Any],
    metrics: Dict[str, Any] = None,
) -> None
```
完成分叉节点的执行。

##### 树可视化与详情

```python
def visualize_tree(self, show_full: bool = False) -> str
```
返回树的字符串表示，用于 CLI 展示。

```python
def get_node_detail(self, node_id: str) -> str
```
获取节点详情（格式化字符串），包含节点信息、执行动作、结果、指标等。

```python
def get_path_detail(self, node_id: str) -> str
```
获取从 root 到指定节点的路径详情（格式化字符串），包含路径上的所有节点和累计指标。

```python
def get_resume_info(self) -> Dict[str, Any]
```
获取任务恢复所需的信息，包含当前节点、状态、动作历史、pending tools、thinking 等。

##### Trace 属性

```python
@property
def trace(self) -> TraceTree
```
获取 Trace 树对象（只读属性）。首次访问时自动加载。

---

### 3. 数据模型

#### 辅助数据模型

```python
@dataclass
class AgentInfo:
    agent_name: str
    agent_id: str = ""

@dataclass
class TaskInfo:
    task_id: str
    task_input: str = ""
    op_name: str = ""
    dsl: str = ""
    backend: str = ""
    arch: str = ""

@dataclass
class ExecutionInfo:
    tool_call_counter: int = 0
    first_thinking_done: bool = False
    current_turn: int = 0

@dataclass
class FileInfo:
    size: int
    last_modified: str
    checksum: str = ""

@dataclass
class Metrics:
    token_used: int = 0
    duration_ms: int = 0
    performance: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionRecord:
    turn: int
    thinking: str
    decision: str
    timestamp: str = field(default_factory=_get_timestamp)

@dataclass
class PlanState:
    goal: str = ""
    steps: List[str] = field(default_factory=list)
    current_step: int = 0
    status: str = "pending"  # "pending", "in_progress", "completed"
```

#### NodeState

```python
@dataclass
class NodeState:
    node_id: str
    turn: int
    status: str  # "init", "running", "completed", "failed"
    agent_info: Dict[str, Any] = field(default_factory=dict)
    task_info: Dict[str, Any] = field(default_factory=dict)
    execution_info: Dict[str, Any] = field(default_factory=dict)
    file_state: Dict[str, Dict] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_get_timestamp)
```

#### ActionRecord

```python
@dataclass
class ActionRecord:
    action_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_get_timestamp)
    duration_ms: Optional[int] = None
    compressed: bool = False
```

#### ThinkingState

```python
@dataclass
class ThinkingState:
    node_id: str
    turn: int
    current_plan: Dict[str, Any] = field(default_factory=dict)
    latest_thinking: str = ""
    decision_history: List[Dict] = field(default_factory=list)  # 存储 DecisionRecord.to_dict() 的结果
    timestamp: str = field(default_factory=_get_timestamp)
    
    def add_decision(self, thinking: str, decision: str) -> None
```
```

#### PendingTool / PendingToolsState

```python
@dataclass
class PendingTool:
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # "pending", "executing", "completed", "failed"
    created_at: str = field(default_factory=_get_timestamp)
    completed_at: Optional[str] = None

@dataclass
class PendingToolsState:
    node_id: str
    turn: int
    pending_tools: List[PendingTool] = field(default_factory=list)
    last_updated: str = field(default_factory=_get_timestamp)
    
    def add_pending_tool(self, tool: PendingTool) -> None
    def mark_completed(self, tool_call_id: str) -> bool
    def get_pending(self) -> List[PendingTool]
```

#### ActionHistoryFact

```python
@dataclass
class ActionHistoryFact:
    node_id: str
    parent_node_id: Optional[str]
    turn: int
    actions: List[ActionRecord] = field(default_factory=list)
    actions_count: int = 0
    last_updated: str = field(default_factory=_get_timestamp)
    
    def add_action(self, action: ActionRecord) -> None
```

#### ActionHistoryCompressed

```python
@dataclass
class ActionHistoryCompressed:
    node_id: str
    turn: int
    actions: List[ActionRecord] = field(default_factory=list)
    total_actions: int = 0
    compressed_count: int = 0
    source_path: List[str] = field(default_factory=list)  # 用于缓存校验的完整路径
    last_updated: str = field(default_factory=_get_timestamp)
    
    def is_valid(self, current_path: List[str]) -> bool
```

压缩版动作历史，用于 LLM 上下文渲染及展示（减少 token 消耗）。包含 `source_path` 以验证缓存有效性。

#### TraceTree / TraceNode

```python
@dataclass
class TraceNode:
    node_id: str
    parent_id: Optional[str]
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    action: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=_get_timestamp)
    children: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpointer_thread: Optional[str] = None

@dataclass
class TraceTree:
    task_id: str
    created_at: str = field(default_factory=_get_timestamp)
    current_node: str = "root"
    tree: Dict[str, TraceNode] = field(default_factory=dict)
    
    def add_node(self, node: TraceNode) -> None
    def get_node(self, node_id: str) -> Optional[TraceNode]
    def set_current_node(self, node_id: str) -> None
    def get_path_to_node(self, node_id: str) -> List[str]
    def get_all_leaf_nodes(self) -> List[str]
```

**注意**：`TraceTree` 没有 `root_id` 字段，root 节点固定为 `"root"`。

---

### 4. 异常类

```python
class FileSystemStateError(Exception):
    """文件系统状态错误基类"""

class NodeNotFoundError(FileSystemStateError):
    """节点未找到"""
    def __init__(self, node_id: str, message: str = None):
        self.node_id = node_id
        if message is None:
            message = f"Node '{node_id}' not found"
        super().__init__(message)

class TraceSystemError(Exception):
    """Trace 系统错误基类"""

class InvalidNodeStateError(FileSystemStateError):
    """无效的节点状态"""
    def __init__(self, node_id: str, reason: str):
        self.node_id = node_id
        self.reason = reason
        super().__init__(f"Invalid state for node '{node_id}': {reason}")

class TraceNotInitializedError(TraceSystemError):
    """Trace 未初始化"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Trace system for task '{task_id}' is not initialized")

class TraceAlreadyExistsError(TraceSystemError):
    """Trace 已存在"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Trace for task '{task_id}' already exists")
```

---

## 使用场景示例

### 场景 1：单次生成流程

```python
from akg_agents.core_v2.filesystem import (
    FileSystemState, TraceSystem, ActionRecord
)

# 初始化
fs = FileSystemState("softmax_001")
trace = TraceSystem("softmax_001")
fs.initialize_task()
trace.initialize()

# Designer 阶段
designer_node = trace.add_node(
    action={"type": "designer", "params": {}},
    result={"success": True, "design": "..."},
)
fs.save_code_file(designer_node, "design.py", design_code)

# Coder 阶段
coder_node = trace.add_node(
    action={"type": "coder", "params": {}},
    result={"success": True, "code_len": 500},
)
fs.save_code_file(coder_node, "kernel.py", kernel_code)

# Verifier 阶段
trace.mark_node_completed(coder_node, metrics={"performance": 1.5})
```

### 场景 2：多轮迭代（evolve）

```python
for round_idx in range(max_rounds):
    # 创建并行分叉
    fork_nodes = trace.create_parallel_forks(
        n=parallel_num,
        action_template={"type": "evolve_round", "round": round_idx},
    )
    
    # 执行每个分叉
    for fork_node in fork_nodes:
        result = await execute_task(...)
        trace.complete_fork(fork_node, result=result, metrics={"perf": result["perf"]})
    
    # 获取最佳结果
    best = trace.get_best_leaf_node(metric="perf")
    trace.switch_node(best)
```

### 场景 3：用户交互分叉

```python
# 用户对方案不满意，创建新分支
trace.switch_node(previous_node)
new_branch = trace.add_node(
    action={"type": "user_feedback", "feedback": "try another approach"},
    result={},
)

# 继续在新分支上工作
...

# 对比两个方案
comparison = trace.compare_nodes(original_result, new_result)
print(f"分叉点: {comparison['fork_point']}")
print(f"方案1性能: {comparison['metrics_1']['performance']}")
print(f"方案2性能: {comparison['metrics_2']['performance']}")
```

### 场景 4：断点恢复

```python
# 程序重启后
fs = FileSystemState("my_task_001")
trace = TraceSystem("my_task_001")

# initialize() 会自动检测已存在的 trace 并加载
trace.initialize()

# 获取当前状态
current_node = trace.get_current_node()
history = trace.get_full_action_history(current_node)
print(f"恢复到节点 {current_node}，已完成 {len(history)} 个动作")

# 或使用 get_resume_info() 获取完整恢复信息
resume_info = trace.get_resume_info()
print(f"任务: {resume_info['task_id']}")
print(f"当前节点: {resume_info['current_node']}")
print(f"待处理工具: {len(resume_info['pending_tools'].get_pending())} 个")

# 继续执行
...
```

### 场景 5：HITL Pending Tools

```python
from akg_agents.core_v2.filesystem import PendingTool, PendingToolsState

# Agent 需要用户确认
pending = PendingToolsState(node_id=current_node, turn=1)
pending.add_pending_tool(PendingTool(
    tool_call_id="confirm_001",
    tool_name="ask_user",
    arguments={"question": "确认使用此方案?"},
))
fs.save_pending_tools(current_node, pending)

# 用户确认后
fs.mark_tool_completed(current_node, "confirm_001")

# 检查是否还有待处理
pending = fs.load_pending_tools(current_node)
has_pending = any(t.status == "pending" for t in pending.pending_tools)
```

---

## 单元测试

### 测试文件位置

```
core_v2/tests/
├── test_filesystem_state.py    # FileSystemState 单元测试 (31 个测试)
├── test_trace_system.py        # TraceSystem 单元测试 (34 个测试)
├── test_compression.py         # ActionCompressor & Compression 单元测试
├── test_cli_trace.py           # CLI trace 命令测试 (9 个测试)
└── integration/
    ├── test_single_mode.py     # 单次生成模式集成测试 (11 个测试)
    ├── test_treesearch_mode.py # TreeSearch 模式集成测试 (11 个测试)
    └── test_dialog_mode.py     # 对话模式集成测试 (10 个测试)
```

### 运行测试

```bash
# 进入项目目录
cd akg_agents/python

# 运行所有 filesystem 相关测试
pytest akg_agents/core_v2/tests/ -v

# 只运行 FileSystemState 测试
pytest akg_agents/core_v2/tests/test_filesystem_state.py -v

# 只运行 TraceSystem 测试
pytest akg_agents/core_v2/tests/test_trace_system.py -v

# 运行集成测试
pytest akg_agents/core_v2/tests/integration/ -v

# 运行特定测试类
pytest akg_agents/core_v2/tests/test_trace_system.py::TestTraceSystemFork -v
```

### 测试覆盖的功能

| 测试文件                   | 覆盖功能                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `test_filesystem_state.py` | 初始化、节点状态CRUD、动作历史、Thinking、Pending Tools、代码文件管理               |
| `test_trace_system.py`     | 初始化、节点添加/切换、自动分叉、路径查询、历史重建、节点对比、并行分叉、叶节点查询 |
| `test_single_mode.py`      | coder_only, default, verifier_only, connect_all 工作流集成                          |
| `test_treesearch_mode.py`  | evolve, adaptive_search 多轮迭代、分叉、恢复                                        |
| `test_dialog_mode.py`      | 对话流程、用户交互分叉、HITL pending tools、Thinking 演进                           |
| `test_cli_trace.py`        | /trace show, switch, compare, path, history, leaves, best 命令                      |

---

## 导入方式

```python
# 推荐：从 __init__.py 导入
from akg_agents.core_v2.filesystem import (
    # 核心类
    FileSystemState,
    TraceSystem,
    ActionCompressor,
    
    # 数据模型
    NodeState,
    TraceNode,
    ActionRecord,
    ActionHistoryFact,
    ActionHistoryCompressed,
    ThinkingState,
    PendingTool,
    PendingToolsState,
    TraceTree,
    # 辅助模型
    AgentInfo,
    TaskInfo,
    ExecutionInfo,
    FileInfo,
    Metrics,
    DecisionRecord,
    PlanState,
    
    # 异常
    FileSystemStateError,
    NodeNotFoundError,
    InvalidNodeStateError,
    TraceSystemError,
    TraceNotInitializedError,
    TraceAlreadyExistsError,
)
```

---

## 注意事项

1. **线程安全**：当前实现不是线程安全的。如果需要并发访问，请在调用方加锁。

2. **存储路径**：默认存储在 `~/.akg_agents/conversations/{task_id}/`，可通过 `base_dir` 参数修改。

3. **断点恢复**：`TraceSystem.initialize()` 会自动检测已存在的 trace 并加载，无需额外处理。

4. **文件格式**：所有 JSON 文件使用 UTF-8 编码，`indent=2`，`ensure_ascii=False`。

5. **节点 ID**：自动生成格式为 `node_001`, `node_002`, ...，root 节点固定为 `"root"`。

6. **动作 ID**：自动生成格式为 `action_001`, `action_002`, ...
