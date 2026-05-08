[English Version](../Workflow.md)

# 工作流（LangGraph）

## 1. 概述

工作流模块提供基于 LangGraph 的工作流编排能力。它提供领域无关的基类，可扩展用于任何场景。

核心组件：

- **BaseState** — 通用工作流状态（TypedDict）
- **BaseWorkflow** — 抽象工作流基类
- **BaseLangGraphTask** — 抽象任务执行基类
- **路由工具** — 步数限制和 Agent 重复限制检查
- **WorkflowRegistry** — 工作流注册与发现
- **WorkflowVisualizer** — Mermaid 图表生成

## 2. BaseState

`BaseState` 是一个 TypedDict，定义了任何 LangGraph 工作流的最小状态字段。

```python
class BaseState(TypedDict, total=False):
    # 任务标识
    task_id: str
    task_label: str
    session_id: str

    # 流程控制
    iteration: int
    step_count: int
    max_iterations: int

    # 历史记录（自动累积）
    agent_history: Annotated[List[str], add]

    # 结果
    success: bool
    error_message: Optional[str]
```

### 扩展状态

领域专用工作流应继承 `BaseState`：

```python
class MyDomainState(BaseState, total=False):
    document_content: str
    analysis_result: dict
```

## 3. BaseWorkflow

`BaseWorkflow` 是所有工作流的抽象基类。子类实现 `build_graph()` 定义图结构。

```python
class BaseWorkflow(ABC, Generic[StateType]):
    def __init__(self, config: dict, trace=None):
        """
        Args:
            config: 配置字典（包含 max_step 等）
            trace: 可选的 Trace 实例，用于记录执行过程
        """

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """构建工作流图。返回未编译的 StateGraph。"""

    def compile(self):
        """编译图。返回可直接调用 ainvoke() 的 LangGraph 应用。"""

    def visualize(self) -> str:
        """生成 Mermaid 格式的流程图。"""
```

### 示例

```python
from langgraph.graph import StateGraph, END
from akg_agents.core_v2.langgraph_base import BaseWorkflow, BaseState

class MyWorkflow(BaseWorkflow[BaseState]):
    def __init__(self, agents: dict, config: dict):
        super().__init__(config)
        self.agents = agents

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(BaseState)
        workflow.add_node("agent_a", self.agents["a"].run)
        workflow.add_node("agent_b", self.agents["b"].run)
        workflow.add_edge("agent_a", "agent_b")
        workflow.add_edge("agent_b", END)
        workflow.set_entry_point("agent_a")
        return workflow
```

## 4. BaseLangGraphTask

`BaseLangGraphTask` 提供任务执行框架。子类实现工作流初始化和状态准备。

```python
class BaseLangGraphTask(ABC):
    def __init__(self, task_id: str, config: dict, workflow_name: str = "default"):

    @abstractmethod
    def _init_workflow(self):
        """初始化 self.workflow 和 self.app。"""

    @abstractmethod
    def _prepare_initial_state(self, init_info: Optional[dict]) -> Dict[str, Any]:
        """返回工作流的初始状态字典。"""

    async def run(self, init_info=None) -> Tuple[bool, dict]:
        """执行任务。返回 (是否成功, 最终状态)。"""
```

### 示例

```python
from akg_agents.core_v2.langgraph_base import BaseLangGraphTask

class MyTask(BaseLangGraphTask):
    def __init__(self, task_id: str, config: dict):
        super().__init__(task_id, config)
        self._init_workflow()

    def _init_workflow(self):
        self.workflow = MyWorkflow(agents={...}, config=self.config)
        self.app = self.workflow.compile()

    def _prepare_initial_state(self, init_info):
        return {
            "task_id": self.task_id,
            "iteration": 0,
            "step_count": 0,
            "agent_history": [],
        }
```

## 5. 路由工具

路由函数用于控制工作流执行流程。

| 函数 | 说明 |
|------|------|
| `check_step_limit(step_count, max_step)` | 步数超限时返回 `True`。 |
| `check_agent_repeat_limit(agent_history, agent_name, max_repeats)` | Agent 连续调用超过 `max_repeats` 次时返回 `True`（默认 3 次）。 |
| `get_illegal_agents(step_count, max_step, agent_history, repeat_limits)` | 返回被禁止的 Agent 名称集合。返回 `{"*"}` 表示全部禁止（步数超限）。 |

### 在路由中使用

```python
from akg_agents.core_v2.langgraph_base import get_illegal_agents

def my_router(state):
    illegal = get_illegal_agents(
        state["step_count"], 20,
        state["agent_history"],
        {"coder": 3, "designer": 2}
    )
    if "*" in illegal:
        return END
    # 路由到下一个 Agent...
```

## 6. WorkflowRegistry

`WorkflowRegistry` 管理工作流类，支持 scope。

```python
from akg_agents.core_v2.workflows import WorkflowRegistry, register_workflow

@register_workflow
class MyWorkflow(BaseWorkflow):
    ...

# 带 scope
@register_workflow(scopes=["op"])
class KernelWorkflow(BaseWorkflow):
    ...
```

### API 参考

| 方法 | 说明 |
|------|------|
| `WorkflowRegistry.register(cls, name, scopes)` | 注册工作流类。 |
| `WorkflowRegistry.get_workflow_class(name)` | 根据名称获取工作流类。 |
| `WorkflowRegistry.list_workflows(scope)` | 列出已注册工作流，可按 scope 过滤。 |
| `WorkflowRegistry.get_tool_config(workflow_name)` | 获取工作流的工具配置（用于 ToolExecutor 分发）。 |
| `WorkflowRegistry.is_registered(name, scope)` | 检查工作流是否已注册（在指定 scope 中）。 |
| `WorkflowRegistry.clear()` | 移除所有已注册工作流。 |

## 7. WorkflowVisualizer

从编译后的工作流生成 Mermaid 图表：

```python
from akg_agents.core_v2.langgraph_base import WorkflowVisualizer

mermaid_str = workflow.visualize()
print(mermaid_str)
```

## 8. 节点追踪

`track_node` 工具用于在工作流中追踪节点执行：

```python
from akg_agents.core_v2.langgraph_base import track_node

# 在工作流节点中使用，追踪执行过程
```
