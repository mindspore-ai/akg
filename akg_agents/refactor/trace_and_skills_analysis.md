# Trace System 与 Skills 系统分析

> 分析 Trace System 与 LangGraph Checkpointer 的关系，以及 Agent + Skills 动态组合实现
> 更新时间: 2026-01-20

---

## 一、Trace System vs LangGraph Checkpointer

### 1.1 两者的核心差异

#### LangGraph Checkpointer
```python
# 现有使用 (core/agent/react_agent.py)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

class MainOpAgent:
    def __init__(self, checkpointer=None, enable_memory=True):
        if enable_memory:
            # Checkpointer 保存对话历史
            self.checkpointer = checkpointer or InMemorySaver()
        
        # 用于 LangGraph 状态管理
        self.app = create_react_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=self.checkpointer  # 传给 LangGraph
        )
```

**Checkpointer 特性**:
- 🔹 **作用**: 保存 LangGraph 的执行状态和对话历史
- 🔹 **存储**: 内存 (InMemorySaver) 或 SQLite (SqliteSaver)
- 🔹 **粒度**: 每个 LangGraph 节点执行后的状态
- 🔹 **用途**: 
  - 断点续跑 (从中断点恢复)
  - 对话历史回溯
  - 状态持久化
- 🔹 **格式**: LangGraph 内部格式 (StateSnapshot)

#### Trace System (新设计)
```python
# 设计目标
class TraceSystem:
    """状态树管理 + 多方案探索"""
    
    def __init__(self, task_id):
        self.root_dir = Path(f"~/.akg/conversations/{task_id}")
        self.tree = {}  # 树结构: node_id -> TraceNode
    
    def create_node(self, parent_id, action, state):
        """创建 Trace 节点"""
        node = TraceNode(
            node_id=uuid.uuid4(),
            parent_id=parent_id,
            state=state,           # 完整状态快照
            action=action,         # Tool/SubAgent/Orchestrator
            result=None,
            timestamp=now(),
            children=[]
        )
        self.save_node_to_file(node)  # 保存到文件系统
        return node
    
    def explore_branch(self, node_id):
        """从某个节点开始新分支 (多方案探索)"""
        pass
    
    def rollback_to(self, node_id):
        """回溯到指定节点"""
        pass
```

**Trace System 特性**:
- 🔸 **作用**: 管理任务执行的状态树 + 支持多方案并行探索
- 🔸 **存储**: 文件系统 (`~/.akg/conversations/{task_id}/`)
- 🔸 **粒度**: 用户可控 (可能是每个主要决策点)
- 🔸 **用途**:
  - 多方案探索 (树的不同分支)
  - 人工回溯和选择
  - 对比不同方案
  - 可视化决策树
- 🔸 **格式**: 自定义 (trace.json)

---

### 1.2 对比表

| 维度 | LangGraph Checkpointer | Trace System |
|------|----------------------|--------------|
| **主要目标** | 对话历史 + 断点续跑 | 状态树 + 多方案探索 |
| **存储位置** | 内存/SQLite | 文件系统 |
| **粒度** | 每个 LangGraph 节点 | 主要决策点 (可控) |
| **树结构** | 线性历史 | 树形结构 (多分支) |
| **持久化** | 可选 (SqliteSaver) | 必须 (文件系统) |
| **可读性** | 内部格式 | 可读的 JSON |
| **多方案** | 不支持 | 核心功能 |
| **回溯** | 支持 (到历史点) | 支持 (到任意节点+分支) |
| **可视化** | 需要额外工具 | 可直接从文件生成 |

---

### 1.3 两者关系

#### 场景1: 都不用 ❌
```python
# 问题: 无状态，无法断点续跑，无法多方案探索
```

#### 场景2: 只用 Checkpointer ❌
```python
# 问题: 
# - 只有线性历史，无法支持多方案并行探索
# - 内部格式，难以可视化和人工分析
```

#### 场景3: 只用 Trace System ⚠️
```python
# 问题:
# - 需要手动管理所有状态持久化
# - LangGraph 的断点续跑能力无法利用
# - 重复造轮子
```

#### 场景4: 两者配合 ✅ (推荐)
```python
# 优势: 各司其职，充分利用两者优势

class KernelAgent(ReActAgent):
    def __init__(self, trace_system, ...):
        # LangGraph Checkpointer: 管理对话历史
        checkpointer = SqliteSaver.from_conn_string(
            f"{trace_system.root_dir}/checkpointer.db"
        )
        super().__init__(checkpointer=checkpointer, ...)
        
        # Trace System: 管理状态树
        self.trace_system = trace_system
    
    async def run(self, task):
        # 1. 创建 Trace 节点 (主要决策点)
        node = self.trace_system.create_node(
            parent_id=self.current_node_id,
            action="start_task",
            state={"task": task}
        )
        
        # 2. LangGraph 执行 (Checkpointer 自动管理历史)
        config = {"configurable": {"thread_id": node.node_id}}
        result = await self.app.ainvoke({"messages": [task]}, config)
        
        # 3. 更新 Trace 节点
        self.trace_system.update_node(node.node_id, result=result)
        
        return result
```

**配合方式**:
- 🔹 **Checkpointer**: 管理 LangGraph 的细粒度状态 (每个节点)
- 🔸 **Trace System**: 管理粗粒度的决策树 (主要决策点)
- 🔹 **Checkpointer**: 保存在 Trace System 的目录下
- 🔸 **Trace System**: 引用 Checkpointer 的 thread_id

---

### 1.4 文件系统结构

```
~/.akg/conversations/{task_id}/
├── trace.json                  # Trace 树结构
├── checkpointer.db             # LangGraph Checkpointer (SQLite)
├── nodes/                      # Trace 节点详细信息
│   ├── {node_id_1}.json
│   ├── {node_id_2}.json
│   └── ...
├── thinking.json               # PlanAgent 固化的计划
├── code/                       # 生成的代码
└── logs/                       # 日志

trace.json 结构:
{
  "task_id": "task_001",
  "root_node": "node_001",
  "nodes": {
    "node_001": {
      "node_id": "node_001",
      "parent_id": null,
      "action": "start_task",
      "checkpointer_thread": "node_001",  # 关联 Checkpointer
      "children": ["node_002", "node_003"],  # 多分支
      "timestamp": "2026-01-20T10:00:00",
      "status": "completed"
    },
    "node_002": {
      "node_id": "node_002",
      "parent_id": "node_001",
      "action": "call_orchestrator:standard",
      "checkpointer_thread": "node_002",
      "children": ["node_004"],
      "timestamp": "2026-01-20T10:05:00",
      "status": "completed"
    },
    "node_003": {
      "node_id": "node_003",
      "parent_id": "node_001",
      "action": "call_tool:verifyTool",  # 另一种尝试
      "checkpointer_thread": "node_003",
      "children": [],
      "timestamp": "2026-01-20T10:05:10",
      "status": "failed"
    }
  }
}
```

---

### 1.5 实现方案

#### Trace System 核心代码
```python
# core_v2/state/trace_system.py

class TraceNode:
    """Trace 节点"""
    node_id: str
    parent_id: Optional[str]
    action: str  # "start_task", "call_orchestrator:standard", "call_tool:verifyTool"
    state: dict  # 状态快照
    result: Optional[dict]
    checkpointer_thread: str  # 关联的 Checkpointer thread_id
    children: List[str]
    timestamp: str
    status: str  # "pending", "running", "completed", "failed"


class TraceSystem:
    """状态树管理系统"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.root_dir = Path.home() / ".aikg" / "conversations" / task_id
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_file = self.root_dir / "trace.json"
        self.nodes_dir = self.root_dir / "nodes"
        self.nodes_dir.mkdir(exist_ok=True)
        
        # 加载或初始化
        self.tree = self._load_tree()
    
    def create_node(self, 
                    parent_id: Optional[str],
                    action: str,
                    state: dict) -> TraceNode:
        """创建新节点"""
        node_id = str(uuid.uuid4())
        
        node = TraceNode(
            node_id=node_id,
            parent_id=parent_id,
            action=action,
            state=state,
            result=None,
            checkpointer_thread=node_id,  # 用于 Checkpointer
            children=[],
            timestamp=datetime.now().isoformat(),
            status="pending"
        )
        
        # 保存节点
        self._save_node(node)
        
        # 更新树结构
        self.tree["nodes"][node_id] = node
        if parent_id:
            self.tree["nodes"][parent_id]["children"].append(node_id)
        else:
            self.tree["root_node"] = node_id
        
        self._save_tree()
        return node
    
    def update_node(self, node_id: str, **updates):
        """更新节点"""
        node = self.tree["nodes"][node_id]
        for key, value in updates.items():
            setattr(node, key, value)
        self._save_node(node)
        self._save_tree()
    
    def get_branch_path(self, node_id: str) -> List[TraceNode]:
        """获取从根到指定节点的路径"""
        path = []
        current = self.tree["nodes"][node_id]
        while current:
            path.insert(0, current)
            parent_id = current["parent_id"]
            current = self.tree["nodes"][parent_id] if parent_id else None
        return path
    
    def list_branches(self) -> List[List[TraceNode]]:
        """列出所有分支 (多方案)"""
        # 找到所有叶子节点
        leaves = [n for n in self.tree["nodes"].values() if not n["children"]]
        return [self.get_branch_path(leaf["node_id"]) for leaf in leaves]
    
    def _save_node(self, node: TraceNode):
        """保存节点到文件"""
        node_file = self.nodes_dir / f"{node.node_id}.json"
        with open(node_file, 'w') as f:
            json.dump(node.__dict__, f, indent=2)
    
    def _save_tree(self):
        """保存树结构"""
        with open(self.trace_file, 'w') as f:
            json.dump(self.tree, f, indent=2)
    
    def _load_tree(self) -> dict:
        """加载树结构"""
        if self.trace_file.exists():
            with open(self.trace_file) as f:
                return json.load(f)
        return {"task_id": self.task_id, "root_node": None, "nodes": {}}
```

#### 与 Checkpointer 集成
```python
# ops/agents/kernel_agent.py

class KernelAgent(ReActAgent):
    def __init__(self, task_id: str, ...):
        # 1. 创建 Trace System
        self.trace_system = TraceSystem(task_id)
        
        # 2. Checkpointer 放在 Trace System 目录下
        checkpointer_path = self.trace_system.root_dir / "checkpointer.db"
        checkpointer = SqliteSaver.from_conn_string(str(checkpointer_path))
        
        # 3. 初始化 ReAct Agent
        super().__init__(checkpointer=checkpointer, ...)
        
        self.current_node = None
    
    async def run(self, task: str):
        # 1. 创建 Trace 节点
        self.current_node = self.trace_system.create_node(
            parent_id=self.current_node.node_id if self.current_node else None,
            action="start_task",
            state={"task": task, "step": 0}
        )
        
        # 2. LangGraph 执行 (使用 Checkpointer)
        config = {
            "configurable": {
                "thread_id": self.current_node.node_id  # 关联
            }
        }
        
        try:
            result = await self.app.ainvoke(
                {"messages": [HumanMessage(content=task)]},
                config=config
            )
            
            # 3. 更新 Trace 节点
            self.trace_system.update_node(
                self.current_node.node_id,
                result=result,
                status="completed"
            )
            
            return result
            
        except Exception as e:
            self.trace_system.update_node(
                self.current_node.node_id,
                status="failed",
                error=str(e)
            )
            raise
    
    async def explore_alternative(self, parent_node_id: str, alternative_action: str):
        """从某个节点开始探索新方案"""
        # 创建新分支
        new_node = self.trace_system.create_node(
            parent_id=parent_node_id,
            action=alternative_action,
            state=self.trace_system.tree["nodes"][parent_node_id]["state"]
        )
        
        # 从新分支开始执行
        self.current_node = new_node
        # ... 执行新方案
```

---

### 1.6 结论

**推荐方案**: ✅ **两者配合使用**

#### 核心原则 ⭐
```
LangGraph Checkpointer: 管理 Workflow 内部 (Orchestrator 内部状态)
Trace System:          管理 Workflow 外部 (Agent 决策树和多方案)
```

**职责划分**:

- **LangGraph Checkpointer** (Workflow 内部):
  - 管理 Orchestrator 执行时的内部状态
  - Designer → Coder → Verifier 节点间的状态流转
  - 对话历史和断点续跑
  - 保存: `checkpointer.db`

- **Trace System** (Workflow 外部):
  - 管理 KernelAgent 的决策树
  - 记录每次调用 Tool/SubAgent/Orchestrator 的决策点
  - 支持多方案并行探索 (不同决策 → 不同分支)
  - 可视化和人工分析
  - 保存: `trace.json` + `nodes/`

**示意图**:
```
KernelAgent (Trace System 管理)
    ↓ 决策1: 调用 Orchestrator
    ├─ StandardOrchestrator (LangGraph Checkpointer 管理)
    │     ├─ Designer Node
    │     ├─ Coder Node  
    │     └─ Verify Node
    │
    ↓ 决策2: 失败,调用 Tool
    └─ verifyTool (Trace System 管理)

每个决策点 = Trace 节点
Orchestrator 内部 = Checkpointer 管理
```

**不冲突，互补！**

---

## 二、Agent + Skills 动态组合实现

### 2.1 设计目标

**核心理念**: Agent 不是写死的类，而是 SYSTEM_PROMPT + Skills 的动态组合

```python
# 不是这样 ❌
class DesignerAgent(AgentBase):
    def __init__(self):
        self.system_prompt = "你是一个算子设计专家..."
        # 硬编码

# 而是这样 ✅
designer = create_agent(
    base_prompt="你是一个算子设计专家",
    skills=[
        "optimization_patterns",  # 优化模式
        "triton_basics",          # Triton 基础
        "cuda_memory",            # CUDA 内存管理
    ]
)
```

---

### 2.2 Skills 结构设计

#### Skills 文件组织
```
ops/skills/
├── base/                       # 基础 Skills
│   ├── agent_role.md           # Agent 角色定义
│   └── interaction.md          # 交互规范
│
├── compilers/                  # 编译器知识
│   ├── cuda_basics.md
│   ├── cuda_advanced.md
│   ├── ascend_basics.md
│   └── ...
│
├── dsls/                       # DSL 知识
│   ├── triton_basics.md
│   ├── triton_advanced.md
│   ├── swft_basics.md
│   └── ...
│
├── patterns/                   # 优化模式
│   ├── memory_coalescing.md
│   ├── loop_unrolling.md
│   ├── shared_memory.md
│   └── ...
│
├── examples/                   # 示例代码
│   ├── matmul_triton.md
│   ├── attention_cuda.md
│   └── ...
│
└── best_practices/             # 最佳实践
    ├── performance_tuning.md
    ├── debugging.md
    └── ...
```

#### Skill 文件格式 (Markdown)
```markdown
<!-- ops/skills/patterns/memory_coalescing.md -->

# Memory Coalescing 优化模式

## 概述
内存合并访问是 GPU 性能优化的关键技术...

## 适用场景
- 连续内存访问模式
- 全局内存读写密集型算子
- ...

## 实现要点
1. 确保线程访问连续地址
2. 使用向量化加载 (float4)
3. ...

## 示例代码
```python
@triton.jit
def coalesced_kernel(...):
    # 合并访问
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    data = tl.load(ptr + offsets)  # 连续访问
    ...
```

## 性能提升
- 理论加速比: 3-5x
- 适用硬件: A100, 910B
```

---

### 2.3 Skills 加载和组合

#### SkillLoader 实现
```python
# core_v2/skills/loader.py

class Skill:
    """单个 Skill"""
    def __init__(self, name: str, content: str, metadata: dict):
        self.name = name
        self.content = content
        self.metadata = metadata  # 场景、硬件、难度等


class SkillLoader:
    """Skills 动态加载器"""
    
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._cache: Dict[str, Skill] = {}
    
    def load_skill(self, skill_path: str) -> Skill:
        """加载单个 Skill"""
        if skill_path in self._cache:
            return self._cache[skill_path]
        
        full_path = self.skills_dir / skill_path
        with open(full_path) as f:
            content = f.read()
        
        # 解析 metadata (YAML front matter)
        metadata = self._parse_metadata(content)
        
        skill = Skill(
            name=skill_path,
            content=content,
            metadata=metadata
        )
        self._cache[skill_path] = skill
        return skill
    
    def load_skills(self, skill_paths: List[str]) -> List[Skill]:
        """批量加载 Skills"""
        return [self.load_skill(path) for path in skill_paths]
    
    def find_skills_by_tag(self, tag: str) -> List[Skill]:
        """按标签查找 Skills"""
        # 遍历 skills 目录，匹配 metadata
        pass
    
    def _parse_metadata(self, content: str) -> dict:
        """解析 Markdown front matter"""
        # YAML front matter 解析
        pass


# ops/skills/manager.py (算子专用)
class KernelSkillManager:
    """算子 Skills 管理器"""
    
    def __init__(self):
        self.loader = SkillLoader(Path("op/skills"))
    
    def get_skills_for_task(self, task_info: dict) -> List[Skill]:
        """根据任务信息动态选择 Skills"""
        dsl = task_info.get("dsl")  # triton_cuda, triton_ascend, ...
        backend = task_info.get("backend")  # cuda, ascend
        complexity = task_info.get("complexity", "basic")  # basic, advanced
        
        skills = []
        
        # 1. 基础 Skills (必选)
        skills.extend(self.loader.load_skills([
            "base/agent_role.md",
            "base/interaction.md"
        ]))
        
        # 2. DSL 相关 (根据 dsl 选择)
        if dsl == "triton_cuda":
            skills.append(self.loader.load_skill("dsls/triton_basics.md"))
            if complexity == "advanced":
                skills.append(self.loader.load_skill("dsls/triton_advanced.md"))
        
        # 3. 编译器相关 (根据 backend 选择)
        if backend == "cuda":
            skills.append(self.loader.load_skill("compilers/cuda_basics.md"))
        elif backend == "ascend":
            skills.append(self.loader.load_skill("compilers/ascend_basics.md"))
        
        # 4. 优化模式 (可选)
        if "optimization" in task_info:
            pattern = task_info["optimization"]
            skills.append(self.loader.load_skill(f"patterns/{pattern}.md"))
        
        return skills
```

---

### 2.4 Agent 动态组合

#### Agent Factory
```python
# ops/agents/factory.py

class KernelAgentFactory:
    """KernelAgent 工厂"""
    
    def __init__(self, skill_manager: KernelSkillManager):
        self.skill_manager = skill_manager
    
    def create_agent(self, 
                     agent_type: str,
                     task_info: dict,
                     **kwargs) -> AgentBase:
        """动态创建 Agent"""
        
        # 1. 加载 Skills
        skills = self.skill_manager.get_skills_for_task(task_info)
        
        # 2. 构建 SYSTEM_PROMPT
        system_prompt = self._build_system_prompt(agent_type, skills)
        
        # 3. 创建 Agent
        if agent_type == "kernel":
            return KernelAgent(
                system_prompt=system_prompt,
                skills=skills,
                **kwargs
            )
        elif agent_type == "designer":
            return DesignerSubAgent(
                system_prompt=system_prompt,
                skills=skills,
                **kwargs
            )
        # ...
    
    def _build_system_prompt(self, 
                            agent_type: str, 
                            skills: List[Skill]) -> str:
        """从 Skills 构建 SYSTEM_PROMPT"""
        
        # 基础模板
        base_template = self._get_base_template(agent_type)
        
        # 组合 Skills
        skills_content = "\n\n".join([
            f"## {skill.name}\n{skill.content}"
            for skill in skills
        ])
        
        # 最终 Prompt
        return f"""{base_template}

# 知识库 (Skills)

{skills_content}

# 指令
请基于上述知识库，按照你的角色职责完成任务。
"""
    
    def _get_base_template(self, agent_type: str) -> str:
        """获取 Agent 基础模板"""
        templates = {
            "kernel": "你是一个算子生成专家，负责...",
            "designer": "你是一个算子设计专家，负责分析需求并设计方案...",
            "coder": "你是一个代码生成专家，负责将设计方案转换为高质量代码...",
        }
        return templates.get(agent_type, "")
```

#### 使用示例
```python
# 创建 KernelAgent

task_info = {
    "op_name": "matmul",
    "dsl": "triton_cuda",
    "backend": "cuda",
    "arch": "a100",
    "complexity": "advanced",
    "optimization": "memory_coalescing"
}

# 1. 初始化 Skill Manager
skill_manager = KernelSkillManager()

# 2. 使用 Factory 创建 Agent
factory = KernelAgentFactory(skill_manager)
kernel_agent = factory.create_agent(
    agent_type="kernel",
    task_info=task_info,
    llm_client=llm_client,
    tools=[verifyTool, compileTool, ...],
    sub_agents=[designer, coder, ...]
)

# 3. Agent 已经包含了动态选择的 Skills
# - base/agent_role.md
# - base/interaction.md
# - dsls/triton_basics.md
# - dsls/triton_advanced.md
# - compilers/cuda_basics.md
# - patterns/memory_coalescing.md

# 4. 运行
result = await kernel_agent.run(task_info)
```

---

### 2.5 Skills 动态更新

#### 运行时加载新 Skills
```python
class KernelAgent(ReActAgent):
    def __init__(self, system_prompt, skills, skill_manager, ...):
        super().__init__(...)
        self.system_prompt = system_prompt
        self.skills = skills
        self.skill_manager = skill_manager
    
    async def load_additional_skill(self, skill_path: str):
        """运行时加载额外 Skill"""
        new_skill = self.skill_manager.load_skill(skill_path)
        self.skills.append(new_skill)
        
        # 重新构建 SYSTEM_PROMPT
        self.system_prompt = self._rebuild_prompt()
        
        # 更新 LLM 上下文
        # (具体实现取决于 LangGraph 机制)
```

---

### 2.6 优势

**对比硬编码 Agent**:

| 方面 | 硬编码 Agent | Skills 动态组合 |
|------|-------------|----------------|
| **灵活性** | ❌ 修改需要改代码 | ✅ 修改 Markdown 即可 |
| **可扩展性** | ❌ 添加知识需要改类 | ✅ 添加新 Skill 文件 |
| **可维护性** | ❌ 知识分散在代码中 | ✅ 知识集中在 Skills |
| **可复用性** | ❌ Agent 间难以共享 | ✅ Skills 可组合复用 |
| **调试性** | ❌ Prompt 埋在代码里 | ✅ 独立文件易查看 |
| **版本控制** | ❌ 代码和知识耦合 | ✅ Skills 独立版本控制 |

---

## 三、总结

### Trace System vs Checkpointer
- ✅ **两者配合使用**
- Checkpointer: 管理 LangGraph 细粒度状态 + 对话历史
- Trace System: 管理粗粒度决策树 + 多方案探索
- 保存在同一目录，通过 thread_id 关联

### Agent + Skills
- ✅ **动态组合模式**
- Agent = SYSTEM_PROMPT Template + Skills (动态加载)
- Skills 以 Markdown 组织，按需加载和组合
- 灵活、可扩展、易维护

---

**更新时间**: 2026-01-20
**状态**: 设计完成，待实现
