# AIKG 现有代码分析 与 重构对比

> 分析现有代码结构,对比重构方案的差异
> 更新时间: 2026-01-20

## 一、现有架构总览

### 1.1 目录结构
```
akg_agents/
├── core/                          # 现有核心代码
│   ├── agent/                     # Agent 实现
│   │   ├── agent_base.py          # AgentBase (600+ 行)
│   │   ├── agent_base_v2.py       # AgentBaseV2
│   │   ├── designer.py            # 设计 Agent
│   │   ├── coder.py               # 编码 Agent
│   │   ├── react_agent.py         # ReAct Agent
│   │   ├── main_op_agent.py       # 主算子 Agent
│   │   └── op_task_builder.py     # 任务构建 Agent
│   ├── langgraph_task.py          # LangGraph 任务执行器
│   ├── task.py                    # 原始任务执行器
│   ├── conductor.py               # Conductor 决策器
│   ├── sub_agent_registry.py      # SubAgent 注册表
│   ├── trace.py                   # Trace 日志
│   ├── verifier/                  # 验证器
│   │   └── kernel_verifier.py
│   ├── worker/                    # Worker 系统
│   │   ├── manager.py
│   │   └── ...
│   └── tools/                     # 工具
│       ├── basic_tools.py
│       └── sub_agent_tool.py
│
├── workflows/                     # LangGraph 工作流
│   ├── base_workflow.py           # 基础工作流
│   ├── default_workflow.py        # 默认工作流
│   ├── coder_only_workflow.py
│   ├── verifier_only_workflow.py
│   └── connect_all_workflow.py
│
├── utils/
│   └── langgraph/                 # LangGraph 工具
│       ├── state.py               # KernelGenState
│       ├── nodes.py               # 节点工厂
│       └── routers.py             # 路由工厂
│
└── cli/                           # CLI
```

---

## 二、现有关键组件分析

### 2.1 LangGraph 使用现状

#### ✅ 已实现的能力
```python
# 1. BaseWorkflow - 工作流基类
class BaseWorkflow(ABC):
    def __init__(self, agents, device_pool, trace, config, ...):
        pass
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """子类实现具体图结构"""
        pass
    
    def compile(self):
        graph = self.build_graph()
        return graph.compile()

# 2. DefaultWorkflow - Designer→Coder→Verifier
class DefaultWorkflow(BaseWorkflow):
    def build_graph(self) -> StateGraph:
        workflow = StateGraph(KernelGenState)
        
        # 添加节点
        workflow.add_node("designer", designer_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("conductor", conductor_node)
        
        # 添加边
        workflow.add_edge("designer", "coder")
        workflow.add_edge("coder", "verifier")
        
        # 条件路由
        workflow.add_conditional_edges(
            "verifier",
            verifier_router,
            {"conductor": "conductor", "finish": END}
        )
        
        return workflow

# 3. KernelGenState - 状态定义
class KernelGenState(TypedDict):
    task_info: dict
    design: Optional[str]
    code: Optional[str]
    verification_result: Optional[dict]
    step: int
    max_step: int
    ...
```

#### ✅ 已有的 Conductor 决策
```python
# conductor_node 使用 LLM 分析当前状态并决策下一步
# 但没有固化到文件系统
```

#### ✅ 已有的 ReAct Agent
```python
# core/agent/react_agent.py - MainOpAgent
class MainOpAgent(AgentBaseV2):
    def __init__(..., checkpointer=None, enable_memory=True):
        # 使用 LangGraph checkpointer 保存历史
        # 使用 trim_messages 压缩对话历史
        pass
    
    # 已实现 ReAct 循环:
    # 1. LLM 决策 → tool_calls
    # 2. 执行 tools (sub_agent_tools, basic_tools)
    # 3. 观察结果 → 下一轮
```

**关键发现**: 
- ✅ 已有 LangGraph StateGraph 编排固定流程
- ✅ 已有 Conductor 节点做决策
- ✅ 已有 ReAct Agent 调用 SubAgent Tools
- ❌ 缺少 Plan/RePlan 显式规划
- ❌ 缺少 Trace 树(多方案探索)
- ❌ 缺少文件系统状态管理
- ❌ 缺少每10步固化机制

---

### 2.2 Agent 系统现状

#### AgentBase (现有)
```python
# core/agent/agent_base.py - 600+ 行
class AgentBase:
    def __init__(self, name, agent_type, config, ...):
        # 混合了太多职责:
        # - LLM 调用
        # - Prompt 构建
        # - RAG 检索
        # - 结果解析
        # - 日志记录
        # ...
        pass
    
    def run_llm(self, prompt, ...):
        """直接调用 LLM"""
        pass
    
    # 大量算子专用逻辑混在基类
```

**问题**:
- ❌ 600+ 行,职责不清
- ❌ 算子专用逻辑在基类
- ❌ LLM 调用耦合在 Agent 内部
- ❌ 难以用于非算子场景

#### SubAgent 注册 (现有)
```python
# core/sub_agent_registry.py
class SubAgentRegistry:
    """全局注册表,管理所有 SubAgent"""
    
    @classmethod
    def register(cls, agent_class):
        cls._registry[agent_class.__name__] = agent_class
    
    @classmethod
    def create_agent(cls, agent_type, **kwargs):
        pass
```

**对比新架构 AgentRegistry**:
- ✅ 已有注册机制
- ❌ 没有装饰器注册 (@register_agent)
- ❌ 没有 can_handle() 接口
- ✅ 可直接复用和增强

---

### 2.3 Worker 系统现状

#### WorkerManager (现有)
```python
# core/worker/manager.py
class WorkerManager:
    """单例,管理本地和远程 Worker"""
    
    async def register_local_worker(devices, backend, arch):
        """注册本地 Worker"""
        pass
    
    async def register_remote_worker(backend, arch, worker_url):
        """注册远程 Worker"""
        pass
    
    async def select(backend, arch):
        """选择 Worker"""
        pass
    
    async def release(worker):
        """释放 Worker"""
        pass
```

**对比新架构 WorkerPool**:
- ✅ 已有完整实现
- ✅ 支持本地/远程
- ✅ 可直接迁移到 core_v2
- 改进点: 增加负载均衡

---

### 2.4 验证器现状

#### KernelVerifier (现有)
```python
# core/verifier/kernel_verifier.py
class KernelVerifier:
    """算子验证器,包含编译、正确性、性能验证"""
    
    def __init__(self, op_name, framework_code, dsl, backend, arch, ...):
        pass
    
    async def run(self, task_info, current_step):
        """运行验证"""
        # 1. 编译
        # 2. 正确性验证
        # 3. 性能评估
        pass
    
    async def generate_reference_data(self, task_desc, timeout):
        """生成参考数据(跨平台)"""
        pass
```

**对比新架构 VerifierBase**:
- ✅ 已有完整验证逻辑
- ❌ 高度耦合算子场景
- ❌ 没有通用 VerifierBase 接口
- 改进: 提取通用接口到 core_v2/verifiers/base.py

---

### 2.5 Trace 系统现状

#### Trace (现有)
```python
# core/trace.py
class Trace:
    """日志记录,保存到 akg_agents_logs/"""
    
    def __init__(self, op_name, task_id, log_dir):
        self.log_dir = log_dir / op_name / task_id
        pass
    
    def log(self, step, agent_name, content):
        """记录日志"""
        pass
    
    # 只是日志记录,不是状态管理
    # 没有树结构
    # 没有多方案探索
```

**对比新架构 TraceManager**:
- ✅ 已有日志目录结构
- ❌ 没有树结构
- ❌ 没有状态快照
- ❌ 没有回溯能力
- 需要: 完全重写为 TraceManager

---

### 2.6 Tools 系统现状

#### BasicTools (现有)
```python
# core/tools/basic_tools.py
def create_basic_tools():
    """创建基础工具: read_file, write_file, execute_code, ..."""
    return [
        Tool(name="read_file", func=..., description=...),
        Tool(name="write_file", func=..., description=...),
        ...
    ]
```

#### SubAgentTool (现有)
```python
# core/tools/sub_agent_tool.py
def create_sub_agent_tools(registry, ...):
    """将 SubAgent 包装为 Tool"""
    tools = []
    for agent_type in registry.list_agents():
        tool = Tool(
            name=f"call_{agent_type}",
            func=lambda ...: registry.create_and_run(...),
            description=...
        )
        tools.append(tool)
    return tools
```

**对比新架构 Tools 系统**:
- ✅ 已有 Tools 实现
- ✅ 已有 SubAgent → Tool 包装
- ✅ 可直接迁移
- 改进: 统一管理,增加分类

---

## 三、LangGraph 与新架构的融合

### 3.1 现有 LangGraph 能力对照表

| 功能 | 现有实现 | 新架构需求 | 融合方案 |
|------|---------|-----------|---------|
| **固定工作流** | ✅ BaseWorkflow + StateGraph | Orchestrator + Workflow | **复用** StateGraph,封装为 Orchestrator;BaseWorkflow → WorkflowBuilder |
| **状态管理** | ✅ KernelGenState TypedDict | 通用 State | **扩展** KernelGenState → GenericState |
| **节点定义** | ✅ NodeFactory | Agent 节点 | **复用** NodeFactory |
| **条件路由** | ✅ Router 函数 | 决策路由 | **复用** RouterFactory |
| **Conductor** | ✅ conductor_node + LLM | PlanAgent | **融合** Conductor → PlanAgent |
| **Checkpointer** | ✅ InMemorySaver/SqliteSaver | 状态持久化 | **扩展** 到文件系统 |
| **trim_messages** | ✅ 已实现 | 历史压缩 | **复用** |
| **ReAct循环** | ✅ MainOpAgent | Main Agent | **复用** 并增强 |

**结论**: LangGraph 现有能力可以大量复用！

---

### 3.2 融合策略 ✅ 正确理解

#### 核心理念: StateGraph 就是 Orchestrator
```
LangGraph StateGraph = 编排器 (提供 add_node/add_edge API)
   ↓ 封装
Orchestrator (对 StateGraph 的薄封装 + YAML 支持)
   ↓ 使用
WorkflowBuilder (原 BaseWorkflow,改名)
   ↓ 构建
Workflow (已编译的可执行流程)
```

#### 正确方案: Orchestrator 封装 StateGraph
```python
# core_v2/orchestrator/orchestrator.py
class Orchestrator:
    """工作流编排器 (对 StateGraph 的封装)"""
    
    def __init__(self, state_schema):
        self.graph = StateGraph(state_schema)  # 核心: 复用 LangGraph
    
    def add_node(self, name, func, **metadata):
        """添加节点"""
        self.graph.add_node(name, func)
        return self  # 链式调用
    
    def add_edge(self, from_node, to_node):
        """添加边"""
        self.graph.add_edge(from_node, to_node)
        return self
    
    def build(self) -> Workflow:
        """构建工作流 (编译)"""
        return Workflow(self.graph.compile())
    
    @classmethod
    def from_yaml(cls, yaml_path, node_factory):
        """从 YAML 配置构建 (复用现有 YAML 系统)"""
        config = WorkflowManager.load_workflow_config(yaml_path)
        orchestrator = cls(KernelGenState)
        # 根据 YAML 配置添加节点和边
        ...
        return orchestrator


# core_v2/orchestrator/builder.py (原 BaseWorkflow)
class WorkflowBuilder(ABC):
    """工作流构建器 (原 BaseWorkflow,改名)"""
    
    def __init__(self, agents, trace_system, config, ...):
        # 现有 BaseWorkflow 初始化逻辑
        self.agents = agents
        self.trace_system = trace_system
        self.orchestrator = Orchestrator(KernelGenState)
    
    @abstractmethod
    def build(self) -> Workflow:  # 原 build_graph
        """子类实现: 使用 self.orchestrator 编排"""
        pass


# ops/orchestrators/standard_kernel.py (原 DefaultWorkflow)
class StandardKernelWorkflowBuilder(WorkflowBuilder):
    """标准算子生成工作流构建器"""
    
    def build(self) -> Workflow:
        """使用 Orchestrator 编排"""
        # 添加节点 (与原 DefaultWorkflow.build_graph() 逻辑类似)
        self.orchestrator.add_node(
            "designer",
            NodeFactory.create_designer_node(self.agents['designer'], ...)
        )
        self.orchestrator.add_node(
            "coder",
            NodeFactory.create_coder_node(self.agents['coder'], ...)
        )
        self.orchestrator.add_node(
            "verify",
            NodeFactory.create_verify_tool_node(...)  # 注意: Tool 化
        )
        
        # 添加边
        self.orchestrator.add_edge("designer", "coder")
        self.orchestrator.add_edge("coder", "verify")
        
        # 条件边
        self.orchestrator.add_conditional_edge(
            "verify",
            RouterFactory.create_verify_router(),
            {"coder": "coder", "finish": END}
        )
        
        self.orchestrator.set_entry("designer")
        
        # 构建工作流
        return self.orchestrator.build()


# ops/agents/kernel_agent.py
class KernelAgent(ReActAgent):  # 继承现有 MainOpAgent
    """算子生成主 Agent (SYSTEM_PROMPT + Skills 组合)"""
    
    def __init__(self, workflow_builders: dict, skills: dict, ...):
        # Skills 动态组合
        system_prompt = self._build_prompt_from_skills(skills)
        super().__init__(system_prompt=system_prompt, ...)
        
        # 构建所有工作流
        self.workflows = {
            name: builder.build()
            for name, builder in workflow_builders.items()
        }
    
    async def decide_action(self, state):
        """决策: Tool / SubAgent / Workflow"""
        if self.should_use_workflow(state):
            # 选择并调用 Workflow
            workflow_name = self.select_workflow(state)
            workflow = self.workflows[workflow_name]
            
            # Trace System 记录
            node = self.trace_system.create_node(
                action=f"call_workflow:{workflow_name}",
                state=state
            )
            result = await workflow.run(state)
            self.trace_system.update_node(node.node_id, result=result)
            
            return result
        else:
            # 调用 Tool / SubAgent
            return await super().decide_action(state)
```

#### 使用方式
```python
# 方式1: 从 YAML 构建 (复用现有配置)
workflow = Orchestrator.from_yaml(
    "config/default_workflow.yaml",
    node_factory
).build()

# 方式2: 用 Builder (预定义)
builder = StandardKernelWorkflowBuilder(agents, trace_system, ...)
workflow = builder.build()

# 方式3: 手动编排
orchestrator = Orchestrator(KernelGenState)
workflow = orchestrator \
    .add_node("designer", designer_func) \
    .add_node("coder", coder_func) \
    .add_edge("designer", "coder") \
    .set_entry("designer") \
    .build()

# 执行
result = await workflow.run(initial_state)
```

**优势**:
- ✅ 充分复用现有代码 (StateGraph, BaseWorkflow, YAML 系统)
- ✅ 概念清晰: Orchestrator 编排出 Workflow
- ✅ 最小改动: BaseWorkflow 只需改名 + 用 Orchestrator API
- ✅ 灵活性: 支持 YAML 配置 + 代码编排

---

### 3.3 PlanAgent 与 Conductor 的融合

#### 现有 Conductor
```python
# workflows/default_workflow.py
conductor_node = NodeFactory.create_conductor_node(
    trace, config, conductor_template
)

# 功能:
# - 分析当前状态 (design, code, verification_result)
# - LLM 决策下一步 (coder/designer/finish)
# - 没有固化到文件系统
```

#### 新 PlanAgent
```python
# core_v2/state/plan_agent.py

class PlanAgent:
    """规划 Agent (增强版 Conductor)
    
    职责:
    1. 初始规划
    2. 每10步重新规划
    3. 固化计划到文件系统
    """
    
    def __init__(self, llm_client, trace_system):
        self.llm_client = llm_client
        self.trace_system = trace_system
        self.conductor_template = load_conductor_template()
    
    def plan(self, state):
        """初始规划"""
        # 复用 Conductor 的 LLM 决策逻辑
        plan = self._llm_analyze(state)
        
        # 新增: 固化到文件系统
        self.trace_system.save_plan(plan)
        
        return plan
    
    def replan(self, state, step):
        """重新规划 (每10步)"""
        if step % 10 == 0:
            # 1. 读取文件系统状态
            fs_state = self.trace_system.get_current_state()
            
            # 2. LLM 分析进展
            plan = self._llm_analyze({**state, **fs_state})
            
            # 3. 固化更新的计划
            self.trace_system.save_plan(plan)
            
            return plan
    
    def _llm_analyze(self, state):
        """复用现有 Conductor 模板"""
        prompt = self.conductor_template.render(state)
        return self.llm_client.generate(prompt)
```

**融合方案**: PlanAgent = Conductor + 文件系统固化 + 每10步触发

---

## 四、重构对比总结

### 4.1 可复用组件 ✅

| 组件 | 现有位置 | 迁移目标 | 改动程度 |
|------|---------|---------|---------|
| **WorkerManager** | core/worker/ | core_v2/workers/ | 小 (直接迁移) |
| **BasicTools** | core/tools/ | core_v2/tools/ | 小 (直接迁移) |
| **SubAgentRegistry** | core/sub_agent_registry.py | core_v2/agents/registry.py | 中 (增加装饰器) |
| **BaseWorkflow** | workflows/base_workflow.py | core_v2/workflows/ | 中 (改名+增强) |
| **KernelGenState** | utils/langgraph/state.py | core_v2/workflows/ | 小 (扩展) |
| **NodeFactory** | utils/langgraph/nodes.py | core_v2/workflows/ | 小 (直接迁移) |
| **RouterFactory** | utils/langgraph/routers.py | core_v2/workflows/ | 小 (直接迁移) |
| **ReActAgent** | core/agent/react_agent.py | hitl/main_agent.py | 中 (增强) |

---

### 4.2 需要重写组件 ❌

| 组件 | 原因 | 迁移目标 |
|------|------|---------|
| **AgentBase** | 600+行,职责混乱 | core_v2/agents/base.py (100行) |
| **Trace** | 只是日志,无状态树 | core_v2/state/trace_system.py (Trace System) |
| **Conductor** | 无固化机制 | core_v2/state/plan_agent.py (PlanAgent) |
| **KernelVerifier** | 算子耦合 | ops/tools/verify_tool.py (verifyTool) |

---

### 4.3 需要新增组件 🆕

| 组件 | 目标位置 | 功能 |
|------|---------|------|
| **Trace System** | core_v2/state/trace_system.py | 状态树管理+多方案探索 |
| **StateStorage** | core_v2/state/storage.py | 文件系统持久化 |
| **PlanAgent** | core_v2/state/plan_agent.py | 规划 + 每10步固化 |
| **ActionCompressor** | core_v2/state/compressor.py | 智能压缩 |
| **LLMProvider** | core_v2/llm/provider.py | LLM 统一接口 |
| **LLMClient** | core_v2/llm/client.py | LLM 客户端 |
| **SkillLoader** | core_v2/skills/loader.py | Skills 动态加载+组合 |
| **Orchestrator** | core_v2/orchestrator/base.py | 固化工作流(复用BaseWorkflow) |
| **verifyTool** | ops/tools/verify_tool.py | 验证工具(原KernelVerifier) |
| **KernelAgent** | ops/agents/kernel_agent.py | 算子主Agent(SYSTEM_PROMPT+Skills) |

---

## 五、迁移难度评估

### 5.1 低难度 (直接迁移) 🟢
- WorkerManager
- BasicTools
- NodeFactory
- RouterFactory
- KernelGenState

**预计时间**: 1-2 天

---

### 5.2 中难度 (适配增强) 🟡
- SubAgentRegistry → AgentRegistry (增加装饰器)
- **Orchestrator** (对 StateGraph 封装,新增 from_yaml)
- **BaseWorkflow → WorkflowBuilder** (改名,使用 Orchestrator API)
- **DefaultWorkflow → StandardKernelWorkflowBuilder** (改名+迁移)
- ReActAgent → KernelAgent (增加 Workflow 调用决策)
- KernelVerifier → verifyTool (Tools 化)

**预计时间**: 1-2 周

---

### 5.3 高难度 (重写) 🔴
- AgentBase (600行 → 100行)
- Trace → TraceManager (完全重写)
- Conductor → PlanAgent (融合+固化)
- 文件系统状态管理 (新增)

**预计时间**: 2-3 周

---

## 六、关键问题与建议

### 6.1 LangGraph 使用建议

#### ✅ 推荐做法
1. **封装 StateGraph**: Orchestrator 薄封装 StateGraph,提供统一 API
2. **复用 BaseWorkflow**: 改名为 WorkflowBuilder,使用 Orchestrator 编排
3. **增强 Conductor**: 改造为 PlanAgent,增加文件系统固化
3. **保持工作流定义**: workflows/ 下的固定流程可直接迁移
4. **利用 Checkpointer**: 扩展到文件系统持久化

#### ❌ 避免做法
1. **不要放弃 LangGraph**: 已有大量投入,不要重写
2. **不要重复造轮子**: StateGraph/Router 都可复用
3. **不要破坏现有 API**: 保持 LangGraphTask 接口兼容

---

### 6.2 迁移优先级建议

#### 阶段1: 基础设施 (1周)
```
1. 创建 core_v2/ 和 ops/ 目录
2. 迁移 WorkerManager (直接复制)
3. 迁移 BasicTools (直接复制)
4. 实现 AgentBase (100行新版)
```

#### 阶段2: 状态管理 (2周)
```
1. 实现 Trace System (状态树管理)
2. 实现 StateStorage (文件系统持久化)
3. 实现 PlanAgent (融合 Conductor + 固化机制)
4. 实现 ActionCompressor (智能压缩)
```

#### 阶段3: LLM & Orchestrator (1周)
```
1. 实现 LLMProvider/LLMClient
2. 复用 BaseWorkflow → core_v2/orchestrator/
3. 增强为 Orchestrator (固化工作流能力)
4. 集成 PlanAgent
```

#### 阶段4: Agent & Tools (2周)
```
1. 增强 SubAgentRegistry → AgentRegistry
2. KernelVerifier → verifyTool (Tools 化)
3. 实现 KernelAgent (SYSTEM_PROMPT + Skills 组合)
4. 迁移算子专用 SubAgent 到 ops/
```

#### 阶段5: Skills & 整合 (2周)
```
1. 实现 Kernel Skills 系统 (编译器、DSL、模式等)
2. Skills 动态加载和组合机制
3. Agent + Skills 集成
4. 端到端测试
```

**总预计**: 8-10 周

---

## 七、风险提示

### 7.1 技术风险
1. **LangGraph 版本兼容**: 确保新版 LangGraph API 兼容
2. **状态序列化**: 文件系统持久化需要仔细设计
3. **性能开销**: Trace 树和状态固化可能增加开销

### 7.2 迁移风险
1. **新旧并存**: core 和 core_v2 长期并存可能混乱
2. **接口兼容**: 需保持 LangGraphTask API 不变
3. **测试覆盖**: 重构后需要大量测试验证

---

**更新时间**: 2026-01-20
**状态**: 分析完成,待讨论
**下一步**: 确认迁移优先级和时间计划
