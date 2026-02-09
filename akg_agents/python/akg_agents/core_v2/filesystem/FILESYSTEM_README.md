# AIKG FileSystem Documentation

## 概述

`core_v2.filesystem` 模块提供基于 **快照文件系统** 的双层状态持久化方案，支持：
- 树状 Trace 管理（分叉、合并、回溯）
- 节点状态快照
- 增量动作历史保存
- LLM 历史压缩与缓存
- 断点续跑
- **快照版本控制**：利用文件系统硬链接实现 $O(1)$ 空间分叉和高效版本管理，**无需 Git 依赖**。

### 核心特性
*   **高效快照**: 节点分叉通过硬链接复制文件，实现写时复制，无物理数据复制开销（但在跨文件系统时会自动回退到物理复制）。
*   **双层存储**: 代码与元数据分离。元数据存储在 `state.json`，代码存储在 `code/` 快照目录。
*   **直接读取**: 支持直接从文件系统读取任意节点的代码快照，无需切换工作区。
*   **原子性工作区**: `workspace` 目录始终反映当前节点的完整代码状态。切换节点时自动重置工作区。

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
├── models.py            # 数据模型 (NodeState, ActionRecord 等)
├── exceptions.py        # 异常类
├── state.py             # FileSystemState 类 (Snapshot Backend 核心逻辑)
├── trace_system.py      # TraceSystem 类
├── compressor.py        # ActionCompressor 类 
├── SNAPSHOT_FILESYSTEM.md # 详细设计文档
└── FILESYSTEM_README.md # 本文档
```



---

## 深入了解：节点与存储结构

### 1. 节点命名规则

AIKG 任务中的节点 ID 生成规则旨在保持人类可读性和唯一性：

*   **根节点**: 固定命名为 `"root"`。它是 Trace 树的起点。
*   **普通节点**: 采用自增格式命名，如 `node_001`, `node_002`, `node_003` 等。
    *   系统内部维护一个全局计数器 (`_node_counter`)。
    *   每次添加新节点时，计数器加 1。
    *   这种设计方便在文件系统中按顺序查看节点目录，同时也便于调试。

### 2. 节点目录详细结构

每个节点在文件系统中拥有一个独立的目录（例如 `nodes/node_001/`），其中包含该节点的所有状态和数据。

```text
{base_dir}/conversations/{task_id}/nodes/node_001/
├── state.json                  # [状态快照] 包含节点状态、执行结果、Metrics
├── thinking.json               # [思考过程] 包含 Agent 的思维链、Plan 和决策
├── actions/                    # [动作历史]
│   ├── action_history_fact.json        # 增量完整历史
│   ├── action_history_compressed.json  # 压缩历史缓存
│   └── pending_tools.json              # 待处理工具 (HITL)
├── code/                       # [代码快照] (新)
│   ├── kernel.py               # 此时刻的代码文件（硬链接或副本）
│   └── utils.py
└── system_prompts/             # [系统提示词] 
    └── turn_001.txt            # 该轮次使用的 System Prompt 存档
```

#### 关键文件说明：
- **`state.json`**: 记录节点的核心 metadata，如 status ("completed"/"failed")，turn (轮次)，execution_info 等。
- **`code/`**: 包含该节点时刻的完整代码快照。如果是从父节点继承且未修改的文件，则是指向父节点文件的硬链接（节省空间）。
- **`action_history_fact.json`**: 仅记录在该节点**新产生**的动作（增量）。通过 TraceSystem 回溯父节点链并合并所有 path 上的 fact，即可重建完整历史。
- **`action_history_compressed.json`**: LLM 压缩后的历史摘要。包含 `source_path` 字段用于校验缓存是否过期（例如父节点链发生变化）。

### 3. 理解硬链接与写时复制

Snapshot Filesystem 的核心在于利用**硬链接**实现类似 Git 的高效版本控制。

#### 什么是硬链接？
普通文件复制是"深拷贝"（数据被复制了一份，占用双倍空间）。而硬链接是创建一个新的**文件名**指向同一个**数据块**（不占用额外磁盘空间）。

```text
[Node A] file.py  ----> [Data Block X (Content="v1")] <---- [Node B] file.py
                        (Ref Count = 2)
```

当 `node_b` 是 `node_a` 的子节点时，初始状态下它们共享所有代码文件。

#### 写时复制
当试图修改 `node_b` 的 `file.py` 时，系统会自动执行以下操作：
1.  **Unlink**: 删除 `node_b` 指向 Data Block X 的链接。
2.  **Copy**: 创建一个新的 Data Block Y（内容为 v2）。
3.  **Relink**: 让 `node_b` 的 `file.py` 指向 Data Block Y。

```text
[Node A] file.py  ----> [Data Block X (Content="v1")]
                        (Ref Count = 1)

[Node B] file.py  ----> [Data Block Y (Content="v2")]
                        (Ref Count = 1)
```

这意味着：只有被修改的文件才会占用新空间，未修改的文件永远只有一份物理存储。这使得创建分支（Fork）的成本极低 ($O(1)$) 且节省空间。

#### FAQ: 谁链接到了谁？代码到底存在哪里？

在硬链接系统中，**所有节点都是平等的**，没有"源文件"和"链接文件"的区别。

*   **物理存储**: 文件的真实内容（数据块）存在于磁盘的隐藏区域（由操作系统管理）。
*   **目录入口**: 
    *   `nodes/root/code/main.py` 是指向数据块 A 的一个入口。
    *   `nodes/node_001/code/main.py` 也是指向数据块 A 的一个入口。
*   **删除机制**: 只有当指向数据块 A 的**所有**入口（硬链接）都被删除时，操作系统才会回收数据块 A 的空间。
*   **结论**: 代码分散存储在各个节点的 `code/` 目录下。每个文件都是"真实"的文件，它们只是共享了底层数据而已。删除父节点不会影响子节点的代码，因为子节点持有的链接依然有效。

---

## 类参考

### 1. FileSystemState

文件系统状态管理器，负责节点级别的持久化。

#### 构造函数

```python
FileSystemState(task_id: str, base_dir: str = None)
```

| 参数       | 类型 | 说明                             |
| ---------- | ---- | -------------------------------- |
| `task_id`  | str  | 任务唯一标识                     |
| `base_dir` | str  | 存储根目录，默认 `~/.akg_agents` |

#### 目录结构

```
{base_dir}/conversations/{task_id}/
├── current_node.txt           # 当前节点 ID
├── trace.json                 # Trace 树结构（由 TraceSystem 管理）
├── logs/                      # 日志目录
├── workspace/                 # [当前工作区]
│   ├── kernel.py              # 当前检出的代码（可变）
│   └── ... 
└── nodes/                     # [快照存储]
    ├── root/
    │   ├── state.json
    │   ├── code/              # root 代码快照
    │   └── ...
    └── node_001/
        ├── state.json
        ├── code/              # node_001 代码快照 (硬链接)
        └── ...
```

#### 核心方法

##### 初始化

```python
def initialize_task(self, force: bool = False) -> None
```
初始化任务目录结构，创建 root 节点及其快照目录。

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
```
获取当前活动节点 ID。

```python
def set_current_node(self, node_id: str) -> None
```
设置当前节点。
*   **副作用**: 清空 `workspace` 目录，并从 `nodes/{node_id}/code/` 复制文件到工作区。
*   **注意**: 从快照复制到工作区是物理复制（非硬链接），以防止在工作区的修改影响到已保存的快照。

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
管理 `state.json` 文件。`file_state` 字段现在记录文件的 `path`, `checksum` (MD5) 和 `size`。

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
def save_code_file(self, node_id: str, filename: str, content: str) -> None
```
*   **功能**: 将代码写入 `workspace` 并同步复制到该节点的快照目录 `nodes/{node_id}/code/`。
*   **写时复制**: 写入快照目录前，会检查是否存在硬链接。如果存在，先 `unlink` 再写入新文件，防止修改传播到共享该硬链接的其他节点。
*   **元数据**: 自动计算 MD5 合并更新 `file_state`。

```python
def load_code_file(self, node_id: str, filename: str) -> Optional[str]
```
*   **功能**: 读取指定节点的代码内容。
*   **机制**: 直接读取 `nodes/{node_id}/code/{filename}`。

```python
def list_code_files(self, node_id: str) -> List[str]
```
*   **功能**: 列出指定节点当前版本的所有代码文件（递归遍历快照目录）。

```python
def export_node_code(self, node_id: str, target_path: str) -> None
```
*   **功能**: 将指定节点的完整代码快照导出到目标目录（使用 `shutil.copytree`）。

```python
def diff_file(self, node_a: str, node_b: str, filename: str) -> str
```
*   **功能**: 比较指定文件在两个节点快照中的差异。返回统一 Diff 格式的字符串。

```python
def diff_nodes(self, node_a: str, node_b: str, output_path: Optional[Path] = None) -> Path
```
*   **功能**: 对比两个节点之间的所有代码差异。
*   **存储**: 默认将补丁文件保存至 `workspace/.akg/diffs/{node_a}_to_{node_b}.patch`。

```python
def copy_node_state(from_node: str, to_node: str) -> NodeState
```
*   **功能**: 复制节点状态到新节点。
*   **Code Snapshot**: 递归复制 `code/` 目录。使用系统硬链接 (`os.link`) 复制文件，此时新旧节点指向同一物理磁盘块，实现零成本复制。

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
内部会自动调用 `fs.copy_node_state` 来继承代码快照。

```python
def switch_node(self, node_id: str) -> None
```
切换到指定节点。会自动更新 `TraceTree.current_node` 并调用 `fs.set_current_node(node_id)` 恢复该节点的代码快照到工作区。

```python
def get_node(self, node_id: str) -> TraceNode
```
获取节点信息。返回 `TraceNode` 对象。

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
```

```python
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

##### 节点对比与查询

```python
def compare_nodes(self, node_1: str, node_2: str) -> Dict[str, Any]
```
对比两个节点的路径，返回分叉点和路径差异。

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
从当前节点创建 N 个并行分叉。返回新节点 ID 列表。这些节点将作为当前节点的子节点。

```python
def complete_fork(
    self,
    node_id: str,
    result: Dict[str, Any],
    metrics: Dict[str, Any] = None,
) -> None
```
完成分叉节点的执行。

##### 合并与对比

```python
def find_lca(self, node_id1: str, node_id2: str) -> str
```
*   **功能**: 寻找两个节点的最近公共祖先 (Lowest Common Ancestor)。

```python
def merge_nodes(self, target_node_id: str, source_node_id: str) -> str
```
*   **功能**: 将源节点的更改合并到目标节点（三路合并）。如果有冲突，将在文件中插入冲突标记并标记节点状态。

##### Trace 属性

```python
@property
def trace(self) -> TraceTree
```
获取 Trace 树对象（只读属性）。首次访问时自动加载。

---

### 3. 数据模型

#### Data Classes

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
- `file_state`: 字典，Key为文件名 (例如 `"code/main.py"`)，Value 为 `{"path": "...", "checksum": "...", "size": 100}`。

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
    decision_history: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=_get_timestamp)
```

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

@dataclass
class TraceTree:
    task_id: str
    created_at: str = field(default_factory=_get_timestamp)
    current_node: str = "root"
    tree: Dict[str, TraceNode] = field(default_factory=dict)
```

---

### 4. 异常类

```python
class FileSystemStateError(Exception): ...
class NodeNotFoundError(FileSystemStateError): ...
class InvalidNodeStateError(FileSystemStateError): ...

class TraceSystemError(Exception): ...
class TraceNotInitializedError(TraceSystemError): ...
class TraceAlreadyExistsError(TraceSystemError): ...
```

---

## 使用场景示例

### 场景 1：单次生成流程

```python
from akg_agents.core_v2.filesystem import (
    FileSystemState, TraceSystem
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
# 保存设计代码
fs.save_code_file(designer_node, "design.py", design_code)

# Coder 阶段
coder_node = trace.add_node(
    action={"type": "coder", "params": {}},
    result={"success": True, "code_len": 500},
)
# 保存实现代码（会继承 design.py 并添加 kernel.py）
fs.save_code_file(coder_node, "kernel.py", kernel_code)

# Verifier 阶段
trace.mark_node_completed(coder_node, metrics={"performance": 1.5})
```

### 场景 2：多轮迭代（Tree Search）

```python
for round_idx in range(max_rounds):
    # 创建并行分叉
    fork_nodes = trace.create_parallel_forks(
        n=parallel_num,
        action_template={"type": "evolve_round", "round": round_idx},
    )
    
    # 执行每个分叉
    for fork_node in fork_nodes:
        # 切换到该分叉节点（自动恢复代码快照）
        trace.switch_node(fork_node)
        
        # 执行任务
        result = await execute_task(...)
        
        # 保存结果
        fs.save_code_file(fork_node, "optimized.py", result["code"])
        trace.complete_fork(fork_node, result=result, metrics={"perf": result["perf"]})
    
    # 获取最佳结果节点
    best = trace.get_best_leaf_node(metric="perf")
    # 切换到最佳节点，准备下一轮迭代
    trace.switch_node(best)
```

### 场景 3：用户交互分叉

```python
# 用户对方案不满意，回溯到上一个节点
trace.switch_node(previous_node)

# 创建新分支尝试不同方向
new_branch = trace.add_node(
    action={"type": "user_feedback", "feedback": "try another approach"},
    result={},
)

```

### 场景 4：断点续跑

```python
# 程序崩溃或重启后...
fs = FileSystemState("my_task_001")
trace = TraceSystem("my_task_001")

# 自动检测并加载已有状态
fs.initialize_task()
trace.initialize()

# 获取上次运行的最后状态
resume_info = trace.get_resume_info()
last_node_id = resume_info["current_node_id"]

# 恢复工作区
fs.set_current_node(last_node_id)
print(f"Resumed from node: {last_node_id}")

# 继续执行...
```

### 场景 5：调试与状态检查

得益于 Snapshot Filesystem 的透明性，可以直接使用操作系统工具检查任意节点的状态，无需编写 Python 代码。

```bash
# 查看某个节点的代码快照
ls -l conversations/task_001/nodes/node_005/code/
cat conversations/task_001/nodes/node_005/code/kernel.py

# 比较两个节点的代码差异 (使用 diff 工具)
diff -r conversations/task_001/nodes/node_003/code/ conversations/task_001/nodes/node_005/code/

# 查看节点的执行结果和 Metrics
cat conversations/task_001/nodes/node_005/state.json
```

### 场景 6：代码导出与人工干预

```python
# 将某个节点的完整代码导出到临时目录，进行人工审查或测试
fs.export_node_code("node_015", "/tmp/review_node_015")

# ... 人工修改 /tmp/review_node_015 下的代码 ...

# 将修改后的代码回写到一个新节点（作为人工分支）
new_node_id = trace.add_node(
    action={"type": "human_edit", "reason": "fix bug"},
    result={"success": True}
)


# 读取修改后的文件并保存
modified_code = Path("/tmp/review_node_015/kernel.py").read_text()
fs.save_code_file(new_node_id, "kernel.py", modified_code)
```

### 场景 7：复杂 Agent 循环

展示如何使用 `ThinkingState` 和 `PendingToolsState` 管理复杂的 Agent 思考与执行循环。

```python
# 1. 保存系统提示词
fs.save_system_prompt(node_id, turn=1, prompt="You are a coding expert...")

# 2. 记录思考过程 (思维链)
fs.update_thinking(
    node_id, 
    latest_thinking="I need to list directory contents first.",
    current_plan={"goal": "List files", "steps": ["ls -la"]}
)

# 3. 记录待执行工具
tool_call = PendingTool(
    tool_call_id="call_001", 
    tool_name="list_dir", 
    arguments={"path": "."}
)
fs.add_pending_tool(node_id, tool_call)

# ... 执行工具 ...

# 4. 标记工具完成并记录结果
fs.mark_tool_completed(node_id, "call_001")
trace.add_node(
    action={"type": "tool_output", "output": "..."},
    result={"success": True}
)
```

---

## 测试与稳定性

本项目拥有完善的自动化测试体系，确保底层文件系统的极致稳定与高性能。

### 1. 验证概要
所有核心功能均经过 **93 个自动化测试用例** 的覆盖，包括：
- **Snapshot 核心**: 初始化、Save/Load、工作区重置。
- **硬链接效率**: 验证未修改文件实现 $O(1)$ 分叉，节省磁盘空间。
- **Diff & Merge**: 统一补丁生成、LCA 寻踪、三路合并与冲突检测。
- **集成测试**: 模拟深层树搜索 (50+ 层) 与高并发分支逻辑。

### 2. 运行测试
```bash
# 执行全量测试套件
$env:PYTHONPATH="akg_agents/python"; python -m pytest -v akg_agents/python/akg_agents/core_v2/tests/
```

### 3. 系统保障
- **跨平台一致性**: 自动处理 Windows/Unix 路径分隔符差异。
- **写时复制 (CoW)**: 修改文件前自动断开硬链接，保证节点间代码严格隔离。
- **断点恢复**: 任务崩溃后可秒级加载 `trace.json` 并通过 `switch_node` 恢复完整工作环境。

详细设计请参考: [SNAPSHOT_FILESYSTEM.md](./SNAPSHOT_FILESYSTEM.md)
