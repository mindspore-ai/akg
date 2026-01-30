# AIKG FileSystem State 核心 API 设计

> 关键类、接口、职责定义  
> 更新时间: 2026-01-22  
> **设计原则**: 简单直接，file-centric，参考 infiagent

---

## 一、核心概念与关系

### 1.1 概念定义

```
Trace (对话跟踪)
  ├─ Tree (树结构)
  │   └─ Node (节点) = 1 Turn (1轮对话) ⭐
  │       ├─ NodeState (节点状态快照 - 轻量级)
  │       ├─ Actions (多个动作/工具调用) ⭐
  │       ├─ ThinkingData (规划思考)
  │       └─ Code Files (生成的代码)
  │
  └─ Current Node (当前节点指针)
```

**核心关系**：
- **Trace** 管理一棵树（Tree），树由多个 **Node** 组成
- 每个 **Node** = **1 个 Turn (1轮对话)**，包含：
  - **NodeState**: 该轮的轻量级状态快照（基础信息）
  - **多个 Actions**: 该轮执行的所有工具调用（基于 OpenAI parallel tool calls）⭐
  - **ThinkingData**: PlanAgent 的规划
  - **Code Files**: 生成的代码文件
- **Trace** 维护一个 **Current Node** 指针，指向当前正在工作的节点

**重要说明**：
- **1 Node = 1 Turn**：每个 Node 代表一轮对话（一次 LLM 响应）
- **1 Turn = 多个 Actions**：OpenAI API 支持在一次响应中返回多个 tool calls
- **Node 是检查点**：每个 Node 是一个完整的检查点，包含该轮的所有操作

---

### 1.2 文件系统映射

```
~/.akg/conversations/{task_id}/
├─ trace.json                      # Trace 树结构
├─ current_node.txt                # 当前节点 ID
│
└─ nodes/                          # 所有节点的数据
    ├─ root/                       # root 节点
    │   └─ state.json
    │
    ├─ node_001/                   # 节点 1
    │   ├─ state.json              # NodeState
    │   ├─ thinking.json           # ThinkingData
    │   ├─ actions/
    │   │   ├─ action_history.json          # 压缩版
    │   │   └─ action_history_fact.json     # 增量版（只保存本节点的）
    │   ├─ code/                   # 生成的代码
    │   └─ system_prompts/         # 可选：调试用
    │
    └─ node_002/
        └─ ...
```

---

### 1.3 核心类架构

```
KernelAgent (主 Agent)
    └─ self.trace: Trace  # 只持有一个 Trace 对象！

Trace (对话跟踪管理器)
    ├─ 管理树结构 (trace.json)
    ├─ 维护当前节点 (current_node)
    └─ 操作节点文件 (nodes/{node_id}/*)

数据类：
    ├─ NodeState (节点状态)
    ├─ Action (动作)
    └─ ThinkingData (思考)
```

---

### 1.4 OpenAI Parallel Tool Calls 示例

**OpenAI API 的行为**：

```python
# 一次 LLM 调用
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个算子生成助手"},
        {"role": "user", "content": "实现一个 ReLU kernel"}
    ],
    tools=[
        {"type": "function", "function": {"name": "file_read", ...}},
        {"type": "function", "function": {"name": "call_designer", ...}},
        {"type": "function", "function": {"name": "file_write", ...}}
    ]
)

# LLM 可能在一次响应中返回多个 tool_calls ⭐
response.choices[0].message.tool_calls = [
    ToolCall(id="call_abc1", function=Function(name="file_read", arguments='{"path": "kernel.cu"}')),
    ToolCall(id="call_abc2", function=Function(name="call_designer", arguments='{"task_desc": "ReLU"}')),
    ToolCall(id="call_abc3", function=Function(name="file_write", arguments='{"path": "design.md", "content": "..."}'))
]
```

**我们的处理**：

```
1 次 OpenAI 响应 = 1 个 Turn = 1 个 Node = 多个 Actions

Node_001:
  ├─ Action_001: file_read (对应 call_abc1)
  ├─ Action_002: call_designer (对应 call_abc2)
  └─ Action_003: file_write (对应 call_abc3)
```

**优势**：
- ✅ 符合 OpenAI API 的设计
- ✅ 自然的检查点（每轮对话是一个完整单元）
- ✅ 树结构更扁平、更易理解

---

## 二、核心数据类

### 2.1 NodeState (节点状态快照)

**定义**: 记录一个 node 的轻量级状态快照（参考 infiagent 的设计）

```python
@dataclass
class NodeState:
    """节点状态快照（轻量级）
    
    只记录最基础的状态信息，详细内容在其他文件：
    - Actions → action_history_fact.json
    - Thinking → thinking.json
    - Code → code/
    """
    # 基础信息
    node_id: str                    # 节点 ID
    turn: int                       # 轮次
    status: str                     # "running" | "success" | "failed"
    timestamp: str                  # 时间戳
    
    # 进度信息（参考 infiagent 的 progress tracking）
    progress: Optional[str]         # 进度描述 (如 "设计阶段完成", "30% completed")
    
    # 性能指标
    metrics: Dict[str, Any]         # {"token_used": 1200, "performance": 65, ...}
```

**存储位置**: `nodes/{node_id}/state.json`

**设计理念**（参考 infiagent）：
- ✅ **轻量级**：只记录基础信息，避免冗余
- ✅ **文件分离**：详细内容在各自的专门文件中
- ✅ **易于理解**：字段清晰、含义明确

**为什么不需要这些字段**：
- ❌ `agent_info`: KernelAgent 自己知道身份，不需要存储
- ❌ `task_info`: 整个 trace 的上下文已经包含，不需要每个 node 重复
- ❌ `execution_info`: tool_call_counter 可以从 actions 数量计算
- ❌ `file_state`: 可以从 `code/` 目录扫描得到，不需要冗余存储

**关系**: 一个 Node 有一个 NodeState (1:1)

---

### 2.2 Action (动作/工具调用)

**定义**: 记录一次工具调用及其结果

```python
@dataclass
class Action:
    """动作/工具调用
    
    记录一次工具的执行过程和结果
    """
    action_id: str                  # 动作 ID
    turn: int                       # 所属轮次
    
    tool_name: str                  # 工具名称 (call_designer, file_read 等)
    tool_call_id: Optional[str]     # OpenAI tool call ID
    arguments: Dict[str, Any]       # 工具参数
    
    result: Dict[str, Any]          # 执行结果
    status: str                     # "success" | "failed"
    error: Optional[str]            # 错误信息
    
    timestamp: str                  # 时间戳
    duration_ms: Optional[int]      # 耗时 (毫秒)
    metadata: Dict[str, Any]        # 元数据 (如 token 使用量)
```

**存储位置**: `nodes/{node_id}/actions/action_history_fact.json` (增量保存)

**与 Node 的关系**: 
- **1 个 Node 包含多个 Actions** (1:N 关系) ⭐
- **1 个 Turn 的所有 tool calls** 都存储在同一个 Node 中
- **OpenAI 支持**: 基于 OpenAI API 的 parallel tool calling 特性
- **增量保存**: 每个 Node 只保存在该 Node 执行的 Actions
- **完整历史**: 沿路径回溯收集所有 Node 的 Actions

**示例**：
```
Node_001 (Turn 1):
  ├─ Action_001: file_read("kernel.cu")
  ├─ Action_002: call_designer(task_desc="ReLU")
  └─ Action_003: file_write("design.md", ...)
  
Node_002 (Turn 2):
  ├─ Action_004: call_coder(design="...")
  └─ Action_005: verify_tool(code="...")
```

---

### 2.3 ThinkingData (规划思考)

**定义**: 记录 PlanAgent 的规划和决策

```python
@dataclass
class ThinkingData:
    """PlanAgent 的规划思考
    
    记录规划、决策、待办事项
    """
    node_id: str                    # 所属节点
    turn: int                       # 轮次
    
    current_plan: Dict[str, Any]    # 当前计划
    # {
    #   "goal": "实现 softmax kernel",
    #   "steps": ["设计", "编码", "验证"],
    #   "current_step": 2,
    #   "status": "in_progress"
    # }
    
    latest_thinking: str            # 最新的思考 (自然语言)
    decision_history: List[Dict]    # 决策历史
    timestamp: str                  # 最后更新时间
```

**存储位置**: `nodes/{node_id}/thinking.json`

**关系**: 一个 Node 有一个 ThinkingData

---

## 三、Trace (核心类)

### 3.1 职责

**Trace 是唯一的状态管理类**，负责：

1. **树管理**
   - 管理 trace.json (树结构)
   - 管理节点之间的父子关系
   - 支持节点的创建、切换

2. **当前节点管理**
   - 维护 current_node 指针
   - 提供当前节点的上下文

3. **文件操作**
   - 操作当前节点的所有文件
   - 支持任意节点的文件读取

4. **路径和历史**
   - 提供从 root 到任意节点的路径
   - 提供完整的 action 历史（回溯收集）

---

### 3.2 类定义

```python
class Trace:
    """对话跟踪管理器
    
    统一管理：
    1. trace.json (树结构)
    2. current_node (当前节点)
    3. 所有 nodes/{node_id}/ 下的文件
    
    设计理念：
    - 所有状态持久化到文件系统
    - 当前节点指针 (current_node) 决定操作的目标
    - 默认操作都作用于当前节点
    """
    
    def __init__(self, task_id: str, base_dir: Optional[Path] = None):
        """初始化 Trace
        
        Args:
            task_id: 任务 ID
            base_dir: 基础目录 (默认: ~/.akg/conversations)
        """
        self.task_id = task_id
        self.base_dir = base_dir or Path.home() / ".akg" / "conversations"
        self.task_dir = self.base_dir / task_id
        
        # 核心文件路径
        self.trace_file = self.task_dir / "trace.json"
        self.current_node_file = self.task_dir / "current_node.txt"
        self.nodes_dir = self.task_dir / "nodes"
        
        # 内部状态
        self._trace: Dict[str, Any] = {}  # trace 树
        self._current_node: str = "root"   # 当前节点 ID
```

---

### 3.3 核心 API

#### 3.3.1 初始化和加载

```python
def init_trace(self) -> None:
    """初始化新 trace
    
    说明:
        - 创建目录结构
        - 创建 trace.json (初始化 root 节点)
        - 设置 current_node = "root"
    """
    pass

def load_trace(self) -> None:
    """加载已有 trace
    
    说明:
        - 读取 trace.json
        - 读取 current_node.txt
        - 恢复内部状态
    """
    pass

def save_trace(self) -> None:
    """保存 trace 到磁盘
    
    说明:
        - 保存 trace.json
        - 保存 current_node.txt
        - 原子写入（先写临时文件，再 rename）
    """
    pass
```

---

#### 3.3.2 树管理

```python
def add_node(
    self,
    actions: List[Action],
    state_snapshot: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    parent_id: Optional[str] = None
) -> str:
    """添加新节点（1 Node = 1 Turn）
    
    Args:
        actions: 该轮执行的所有 actions (基于 OpenAI parallel tool calls) ⭐
        state_snapshot: 状态快照 (可选)
        metrics: 指标 (可选) {"token_used": 1200, "performance": 65}
        parent_id: 父节点 ID (可选，默认为当前节点)
        
    Returns:
        新节点的 ID (如 "node_003")
        
    说明:
        - 在父节点下创建新子节点
        - 1 个 Node 包含该 Turn 的所有 Actions
        - 自动检测分叉 (parent 已有子节点)
        - 更新 trace.json (保存 actions 摘要)
        - 切换 current_node 到新节点
        - 创建新节点的目录
        - 从父节点复制 state.json 作为初始状态
        
    示例:
        # 一次 LLM 响应返回 3 个 tool calls
        actions = [
            Action(tool_name="file_read", ...),
            Action(tool_name="call_designer", ...),
            Action(tool_name="file_write", ...)
        ]
        new_node_id = trace.add_node(actions=actions)
    """
    pass

def get_node_info(self, node_id: str) -> Dict[str, Any]:
    """获取节点信息
    
    Args:
        node_id: 节点 ID
        
    Returns:
        {
            "node_id": "node_002",
            "parent_id": "node_001",
            "children": ["node_003", "node_004"],
            "action": {...},
            "result": {...},
            "timestamp": "...",
            "metrics": {...}
        }
    """
    pass

def node_exists(self, node_id: str) -> bool:
    """检查节点是否存在"""
    pass

def get_children(self, node_id: str) -> List[str]:
    """获取节点的所有子节点"""
    pass
```

---

#### 3.3.3 当前节点管理

```python
def get_current_node(self) -> str:
    """获取当前节点 ID
    
    Returns:
        节点 ID (如 "node_003")
    """
    return self._current_node

def switch_node(self, node_id: str) -> None:
    """切换到指定节点
    
    Args:
        node_id: 目标节点 ID
        
    Raises:
        ValueError: 如果节点不存在
        
    说明:
        - 更新 self._current_node
        - 更新 trace.json 中的 current_node
        - 保存到 current_node.txt
    """
    pass
```

---

#### 3.3.4 当前节点的文件操作

**说明**: 以下方法都操作当前节点 (self._current_node)

```python
def save_state(self, state: NodeState) -> None:
    """保存状态到当前节点
    
    Args:
        state: NodeState 对象
        
    文件路径: nodes/{current_node}/state.json
    """
    pass

def load_state(self) -> Optional[NodeState]:
    """从当前节点加载状态
    
    Returns:
        NodeState 对象，如果不存在返回 None
    """
    pass

def get_state_summary(self) -> Dict[str, Any]:
    """获取当前节点的状态摘要
    
    Returns:
        {
            "current_state": NodeState 的精简版,
            "current_plan": thinking 中的 plan,
            "recent_actions": 最近 N 个 actions 的摘要
        }
        
    说明:
        - 用于给 LLM 构建上下文
        - 自动压缩和截断，避免 token 过多
    """
    pass
```

---

```python
def save_action(self, action: Action) -> None:
    """保存 action 到当前节点（增量保存）
    
    Args:
        action: Action 对象
        
    说明:
        - 增量追加到 action_history_fact.json
        - 只保存当前节点新增的 actions
        - 同时更新压缩版 action_history.json
        
    文件路径: nodes/{current_node}/actions/action_history_fact.json
    """
    pass

def get_actions(self) -> List[Action]:
    """获取当前节点的 actions（只包含本节点的）
    
    Returns:
        List[Action] - 当前节点增量保存的 actions
    """
    pass

def get_full_action_history(self) -> List[Action]:
    """获取从 root 到当前节点的完整 action 历史
    
    Returns:
        List[Action] - 完整的 action 列表
        
    说明:
        - 沿路径回溯 (root → ... → current_node)
        - 收集每个节点的 action_history_fact.json
        - 合并并按时间排序
    """
    pass
```

---

```python
def save_thinking(self, thinking: ThinkingData) -> None:
    """保存 thinking 到当前节点
    
    Args:
        thinking: ThinkingData 对象
        
    文件路径: nodes/{current_node}/thinking.json
    """
    pass

def load_thinking(self) -> Optional[ThinkingData]:
    """从当前节点加载 thinking
    
    Returns:
        ThinkingData 对象，如果不存在返回 None
    """
    pass
```

---

```python
def save_code_file(self, relative_path: str, content: str) -> None:
    """保存代码文件到当前节点
    
    Args:
        relative_path: 相对路径 (如 "kernel.cu", "test/test.py")
        content: 文件内容
        
    说明:
        - 保存到 nodes/{current_node}/code/{relative_path}
        - 自动创建父目录
    """
    pass

def load_code_file(self, relative_path: str) -> Optional[str]:
    """从当前节点读取代码文件
    
    Args:
        relative_path: 相对路径
        
    Returns:
        文件内容，如果不存在返回 None
    """
    pass

def list_code_files(self) -> List[str]:
    """列出当前节点的所有代码文件
    
    Returns:
        List[str] - 相对路径列表
    """
    pass

def get_code_file_info(self, relative_path: str) -> Optional[Dict[str, Any]]:
    """获取当前节点的代码文件信息
    
    Returns:
        {
            "path": "kernel.cu",
            "size": 1024,
            "last_modified": "2026-01-22T10:00:00Z",
            "checksum": "abc123..."
        }
    """
    pass
```

---

#### 3.3.5 路径和历史

```python
def get_path_to_node(self, node_id: str) -> List[str]:
    """获取从 root 到指定节点的路径
    
    Args:
        node_id: 目标节点 ID
        
    Returns:
        节点 ID 列表 ["root", "node_001", "node_002", "node_003"]
        
    说明:
        - 沿 parent_id 向上回溯
    """
    pass

def get_path_to_current_node(self) -> List[str]:
    """获取从 root 到当前节点的路径"""
    return self.get_path_to_node(self._current_node)

def get_full_action_history_for_node(self, node_id: str) -> List[Action]:
    """获取从 root 到指定节点的完整 action 历史
    
    Args:
        node_id: 目标节点 ID
        
    Returns:
        List[Action] - 完整的 action 列表
    """
    pass
```

---

#### 3.3.6 可视化和对比

```python
def visualize_tree(self, format: str = "text") -> str:
    """可视化 trace 树
    
    Args:
        format: 格式 ("text" | "json")
        
    Returns:
        树的字符串表示
    """
    pass

def compare_nodes(self, node_1: str, node_2: str) -> Dict[str, Any]:
    """对比两个节点的路径和指标
    
    Args:
        node_1: 节点 1 ID
        node_2: 节点 2 ID
        
    Returns:
        {
            "path_1": ["root", "node_001", "node_003"],
            "path_2": ["root", "node_001", "node_004"],
            "fork_point": "node_001",
            "metrics_1": {"total_token": 3700, "performance": 65},
            "metrics_2": {"total_token": 5000, "performance": 85}
        }
    """
    pass

def get_tree_statistics(self) -> Dict[str, Any]:
    """获取树的统计信息
    
    Returns:
        {
            "total_nodes": 10,
            "max_depth": 5,
            "num_forks": 3,
            "total_token_used": 15000,
            "leaf_nodes": ["node_005", "node_006"]
        }
    """
    pass
```

---

#### 3.3.7 任意节点的文件操作（可选）

**说明**: 以下方法可以操作任意节点（不一定是当前节点）

```python
def save_state_to_node(self, node_id: str, state: NodeState) -> None:
    """保存状态到指定节点"""
    pass

def load_state_from_node(self, node_id: str) -> Optional[NodeState]:
    """从指定节点加载状态"""
    pass

def load_thinking_from_node(self, node_id: str) -> Optional[ThinkingData]:
    """从指定节点加载 thinking"""
    pass
```

---

#### 3.3.8 工具方法

```python
def _get_node_dir(self, node_id: str) -> Path:
    """获取节点目录路径
    
    Args:
        node_id: 节点 ID
        
    Returns:
        Path 对象 (如 ~/.akg/conversations/task_001/nodes/node_003)
    """
    return self.nodes_dir / node_id

def _init_node_directories(self, node_id: str) -> None:
    """初始化节点目录结构
    
    Args:
        node_id: 节点 ID
        
    说明:
        - 创建 nodes/{node_id}/
        - 创建子目录: actions/, code/, system_prompts/
    """
    pass

def _generate_node_id(self) -> str:
    """生成新节点 ID
    
    Returns:
        节点 ID (如 "node_003")
        
    说明:
        - 格式: "node_{counter:03d}"
        - 自动递增
    """
    pass
```

---

## 四、KernelAgent 使用模式

### 4.1 初始化

```python
class KernelAgent:
    def __init__(self, task_id: str):
        self.task_id = task_id
        
        # 只持有一个 Trace 对象！
        self.trace = Trace(task_id)
        
        # 其他组件
        self.plan_agent = PlanAgent(...)
        self.llm_client = LLMClient(...)
        self.tools = [...]
        
        self.current_turn = 0
    
    def init_task(self):
        """初始化新任务"""
        # 初始化 trace
        self.trace.init_trace()
        
        # 当前节点自动为 "root"
        print(f"当前节点: {self.trace.get_current_node()}")  # "root"
    
    def resume_task(self):
        """恢复已有任务"""
        # 加载 trace
        self.trace.load_trace()
        
        # 获取当前节点
        current_node = self.trace.get_current_node()
        print(f"恢复到节点: {current_node}")
        
        # 加载状态
        state = self.trace.load_state()
        if state:
            self.current_turn = state.turn
```

---

### 4.2 执行一轮对话（核心流程）⭐

```python
async def execute_turn(self) -> Dict[str, Any]:
    """执行一轮对话（1 Turn = 1 Node = 多个 Actions）"""
    
    # 1. 构建上下文
    context = self._build_context(
        user_request=self.task_input,
        full_history=self.trace.get_full_action_history(),
        state_summary=self.trace.get_state_summary()
    )
    
    # 2. LLM 决策（可能返回多个 tool calls）
    response = await self.llm_client.chat(
        system_prompt=context,
        tools=self.tools
    )
    
    # 3. 收集该轮的所有 actions
    actions = []
    for tool_call in response.tool_calls:
        # 执行工具
        tool = self._get_tool(tool_call.function.name)
        result = await tool(**json.loads(tool_call.function.arguments))
        
        # 记录 action
        action = Action(
            action_id=f"action_{self.action_counter:03d}",
            turn=self.current_turn,
            tool_name=tool_call.function.name,
            tool_call_id=tool_call.id,
            arguments=json.loads(tool_call.function.arguments),
            result=result,
            status="success" if result.get("success") else "failed",
            timestamp=datetime.now().isoformat(),
            duration_ms=...,
            metadata={}
        )
        actions.append(action)
        self.action_counter += 1
    
    # 4. 创建新节点（包含该轮的所有 actions）⭐
    new_node_id = self.trace.add_node(
        actions=actions,  # 传入列表！
        state_snapshot={"turn": self.current_turn, "status": "running"},
        metrics={"token_used": response.usage.total_tokens}
    )
    
    # 现在 trace.get_current_node() == new_node_id
    
    # 5. 保存所有 actions 到新节点
    for action in actions:
        self.trace.save_action(action)
    
    # 6. 更新轮次
    self.current_turn += 1
    
    return {"node_id": new_node_id, "actions": actions}
```

**说明**：
- **1 次 execute_turn() = 1 个 Node**
- **1 个 Node 包含该 Turn 的所有 Actions**
- 基于 OpenAI API 的 parallel tool calls
- LLM 可以在一次响应中返回多个 tool_calls

---

### 4.3 状态管理

```python
def save_state(self):
    """保存当前状态"""
    state = NodeState(
        node_id=self.trace.get_current_node(),
        turn=self.current_turn,
        status="running",
        timestamp=datetime.now().isoformat(),
        progress="编码阶段进行中",  # 可选的进度描述
        metrics={
            "token_used": self.total_token_used,
            "actions_count": len(self.trace.get_actions())
        }
    )
    
    # 保存到当前节点
    self.trace.save_state(state)

def load_state(self) -> Optional[NodeState]:
    """加载当前节点的状态"""
    return self.trace.load_state()

def get_state_summary(self) -> Dict[str, Any]:
    """获取状态摘要（用于构建 LLM 上下文）"""
    return self.trace.get_state_summary()

def get_full_action_history(self) -> List[Action]:
    """获取完整的 action 历史（从 root 到当前节点）"""
    return self.trace.get_full_action_history()
```

---

### 4.4 节点切换

```python
def switch_node(self, node_id: str):
    """切换到指定节点（用于 /trace switch 命令）"""
    # 切换节点
    self.trace.switch_node(node_id)
    
    # 加载该节点的状态
    state = self.trace.load_state()
    if state:
        self.current_turn = state.turn
        # 恢复其他状态...
    
    print(f"已切换到节点: {node_id}")
```

---

### 4.5 代码文件管理

```python
def save_generated_code(self, file_path: str, content: str):
    """保存生成的代码"""
    # 保存到当前节点
    self.trace.save_code_file(file_path, content)

def load_code(self, file_path: str) -> Optional[str]:
    """加载代码文件"""
    return self.trace.load_code_file(file_path)

def list_all_code_files(self) -> List[str]:
    """列出当前节点的所有代码文件"""
    return self.trace.list_code_files()
```

---

### 4.6 主循环（伪代码）

```python
async def run(self, user_request: str):
    """主循环"""
    # 1. 初始规划
    plan = await self.plan_agent.plan(user_request)
    self.trace.save_thinking(plan)
    
    # 2. 主循环（每轮创建一个 Node）
    for turn in range(100):
        # 执行一轮对话（包含多个 tool calls）
        result = await self.execute_turn()
        
        # 检查是否完成
        for action in result["actions"]:
            if action.tool_name == "finish":
                return action.result
        
        # 保存状态
        self.save_state()
        
        # 每10轮重新规划
        if (turn + 1) % 10 == 0:
            plan = await self.plan_agent.replan(
                user_request=user_request,
                action_history=self.trace.get_full_action_history(),
                current_state=self.trace.get_state_summary()
            )
            self.trace.save_thinking(plan)
```

**循环说明**：
- 每次循环 = 1 个 Turn = 1 个 Node
- 每个 Node 可能包含多个 Actions (parallel tool calls)
- 状态在每轮后保存
- 每 10 轮触发 PlanAgent 重新规划

---

## 五、关键流程示例

### 5.1 创建新任务

```python
# 1. 创建 KernelAgent
agent = KernelAgent(task_id="task_001")

# 2. 初始化任务
agent.init_task()
# 内部: trace.init_trace() → 创建 root 节点 → current_node = "root"

# 3. 现在可以开始执行
print(agent.trace.get_current_node())  # "root"
```

---

### 5.2 执行一轮对话并创建节点 ⭐

```python
# 执行第一轮对话
# LLM 返回 3 个 tool calls: file_read + call_designer + file_write
result = await agent.execute_turn()

# 内部发生了什么：
# 1. LLM 决策 → 返回 3 个 tool_calls
# 2. 执行所有 tools，收集 3 个 Actions
# 3. trace.add_node(actions=[action1, action2, action3])
#    → 创建 node_001，包含 3 个 actions
#    → 设 node_001 为当前节点
# 4. 保存 3 个 actions 到 node_001/actions/action_history_fact.json

print(agent.trace.get_current_node())  # "node_001"
print(len(result["actions"]))  # 3

# node_001 的结构：
# node_001/
#   ├─ state.json
#   ├─ actions/
#   │   └─ action_history_fact.json  # 包含 3 个 actions
#   └─ code/
```

**OpenAI API 行为示例**：

```python
# OpenAI 的一次响应可能包含多个 tool_calls
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tools=[...]
)

# response.choices[0].message.tool_calls 可能是：
[
    ToolCall(id="call_1", function=Function(name="file_read", ...)),
    ToolCall(id="call_2", function=Function(name="call_designer", ...)),
    ToolCall(id="call_3", function=Function(name="file_write", ...))
]

# → 我们创建 1 个 Node，包含 3 个 Actions
```

---

### 5.3 保存和加载状态

```python
# 保存状态
agent.save_state()
# → 保存到 nodes/{current_node}/state.json

# 加载状态
state = agent.load_state()
print(state.turn)  # 当前轮次
print(state.status)  # "running"
```

---

### 5.4 获取完整历史

```python
# 假设路径: root → node_001 → node_002 → node_003
# 每个 node 增量保存了自己的 actions

# 获取完整历史
full_history = agent.get_full_action_history()
# → 自动回溯收集: node_001 的 actions + node_002 的 actions + node_003 的 actions

for action in full_history:
    print(f"[{action.turn}] {action.tool_name}: {action.status}")
```

---

### 5.5 切换节点

```python
# 当前在 node_003

# 切换到 node_002
agent.switch_node("node_002")
# → trace.switch_node("node_002")
# → trace._current_node = "node_002"

# 现在所有操作都作用于 node_002
print(agent.trace.get_current_node())  # "node_002"

# 加载 node_002 的状态
state = agent.load_state()
print(state.turn)
```

---

### 5.6 分叉场景

```python
# 当前在 node_002，已经有一个子节点 node_003

# 再次执行 action（从 node_002）
result = await agent.execute_action(
    tool_name="call_coder",
    tool_args={"strategy": "shared_memory"}
)

# trace.add_node() 检测到 node_002 已有子节点
# → 创建分叉: node_004（node_002 的第二个子节点）
# → current_node = "node_004"

# 现在树结构：
# root → node_001 → node_002
#                      ├─ node_003
#                      └─ node_004 (当前)
```

---

## 六、配置

```python
@dataclass
class TraceConfig:
    """Trace 配置"""
    base_dir: Path = Path.home() / ".akg" / "conversations"
    
    # System Prompt 保存（调试用）
    save_system_prompts: bool = False  # 生产环境关闭
    
    # Action 压缩
    compress_action_results: bool = True
    action_result_max_length: int = 200
    
    # JSON 格式化
    pretty_json: bool = True
    json_indent: int = 2
```

---

## 七、错误处理

### 7.1 自定义异常

```python
class TraceError(Exception):
    """Trace 基础异常"""
    pass

class NodeNotFoundError(TraceError):
    """节点不存在"""
    pass

class TraceCorruptedError(TraceError):
    """Trace 文件损坏"""
    pass

class InvalidNodeStateError(TraceError):
    """节点状态无效"""
    pass
```

---

## 八、总结

### 8.1 核心概念

| 概念 | 说明 | 关系 |
|-----|------|------|
| **Trace** | 对话跟踪管理器 | 管理树、当前节点、所有文件 |
| **Node** | 树的节点 = 1 Turn (1轮对话) ⭐ | 包含 State、Actions、Thinking、Code |
| **Turn** | 一轮对话 | LLM 的一次响应（可能包含多个 tool calls）|
| **NodeState** | 节点状态快照（轻量级）| 一个 Node 有一个 State (1:1) |
| **Action** | 工具调用 | 一个 Node 有多个 Actions (1:N) ⭐ |
| **ThinkingData** | 规划思考 | 一个 Node 有一个 ThinkingData (1:1) |

**核心等式**：
```
1 Node = 1 Turn = 1次 LLM 响应 = 多个 Actions (parallel tool calls)
```

**说明**：
- `task_id` 只是一个标识符，用于区分不同的对话会话
- 不引入 "Task" 作为独立的概念层级，保持简单

---

### 8.2 关键特性

1. **统一管理**: 只有一个 `Trace` 类，管理所有状态
2. **当前节点**: `Trace` 维护 `current_node` 指针，默认操作都作用于当前节点
3. **增量保存**: 每个 Node 只保存自己的 Actions，节省空间
4. **回溯收集**: 需要完整历史时，沿路径回溯收集
5. **自动切换**: `add_node()` 自动切换到新节点
6. **文件为中心**: 所有状态持久化到文件系统

---

### 8.3 使用流程

```
1. 初始化: 
   Trace.init_trace() → 创建 root

2. 执行一轮对话:
   LLM 响应 → 多个 tool_calls → 多个 Actions
   → add_node(actions=[...]) → 创建 1 个 Node → 自动切换

3. 保存状态:
   save_state() → 保存到当前节点
   save_action() → 保存每个 action

4. 获取历史:
   get_full_action_history() → 回溯收集所有 Nodes 的 Actions

5. 切换节点:
   switch_node() → 更新 current_node
```

**典型场景**：
```
Turn 1 (Node_001):
  LLM 返回 3 个 tool_calls
  → 创建 node_001，包含 3 个 actions

Turn 2 (Node_002):
  LLM 返回 2 个 tool_calls
  → 创建 node_002，包含 2 个 actions

Turn 3 (Node_003):
  LLM 返回 1 个 tool_call
  → 创建 node_003，包含 1 个 action
```

---

**设计优势**：
- ✅ **简单**: 只有一个 `Trace` 类
- ✅ **清晰**: 1 Node = 1 Turn = 多个 Actions
- ✅ **易用**: KernelAgent 只需操作 `self.trace`
- ✅ **标准**: 基于 OpenAI parallel tool calls
- ✅ **灵活**: 支持分叉、切换、对比
- ✅ **高效**: 增量保存，节省空间

---

**下一步**: 开始实现代码骨架？
