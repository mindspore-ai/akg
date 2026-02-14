[中文版](./CN/Workflow.md)

# Workflow (LangGraph)

## 1. Overview

The Workflow module provides LangGraph-based workflow orchestration for AKG Agents. It offers domain-agnostic base classes that can be extended for any scenario.

Key components:

- **BaseState** — Generic workflow state (TypedDict)
- **BaseWorkflow** — Abstract workflow base class
- **BaseLangGraphTask** — Abstract task execution base class
- **Router utilities** — Step limit and agent repeat limit checks
- **WorkflowRegistry** — Workflow registration and discovery
- **WorkflowVisualizer** — Mermaid diagram generation

## 2. BaseState

`BaseState` is a TypedDict defining the minimal state fields for any LangGraph workflow.

```python
class BaseState(TypedDict, total=False):
    # Task identification
    task_id: str
    task_label: str
    session_id: str

    # Flow control
    iteration: int
    step_count: int
    max_iterations: int

    # History (auto-accumulating)
    agent_history: Annotated[List[str], add]

    # Results
    success: bool
    error_message: Optional[str]
```

### Extending State

Domain-specific workflows should extend `BaseState`:

```python
class MyDomainState(BaseState, total=False):
    document_content: str
    analysis_result: dict
```

## 3. BaseWorkflow

`BaseWorkflow` is the abstract base class for all workflows. Subclasses implement `build_graph()` to define the graph structure.

```python
class BaseWorkflow(ABC, Generic[StateType]):
    def __init__(self, config: dict, trace=None):
        """
        Args:
            config: Configuration dict (includes max_step, etc.)
            trace: Optional Trace instance for execution recording
        """

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the workflow graph. Returns an uncompiled StateGraph."""

    def compile(self):
        """Compile the graph. Returns a LangGraph app for ainvoke()."""

    def visualize(self) -> str:
        """Generate a Mermaid diagram string."""
```

### Example

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

`BaseLangGraphTask` provides the task execution framework. Subclasses implement workflow initialization and state preparation.

```python
class BaseLangGraphTask(ABC):
    def __init__(self, task_id: str, config: dict, workflow_name: str = "default"):

    @abstractmethod
    def _init_workflow(self):
        """Initialize self.workflow and self.app."""

    @abstractmethod
    def _prepare_initial_state(self, init_info: Optional[dict]) -> Dict[str, Any]:
        """Return the initial state dict for the workflow."""

    async def run(self, init_info=None) -> Tuple[bool, dict]:
        """Execute the task. Returns (success, final_state)."""
```

### Example

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

## 5. Router Utilities

Router functions help control workflow execution flow.

| Function | Description |
|----------|-------------|
| `check_step_limit(step_count, max_step)` | Returns `True` if step count exceeds the limit. |
| `check_agent_repeat_limit(agent_history, agent_name, max_repeats)` | Returns `True` if an agent has been called consecutively more than `max_repeats` times (default: 3). |
| `get_illegal_agents(step_count, max_step, agent_history, repeat_limits)` | Returns a set of forbidden agent names. Returns `{"*"}` if all agents are forbidden (step limit exceeded). |

### Usage in Routers

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
    # Route to next agent...
```

## 6. WorkflowRegistry

`WorkflowRegistry` manages workflow classes with scope support.

```python
from akg_agents.core_v2.workflows import WorkflowRegistry, register_workflow

@register_workflow
class MyWorkflow(BaseWorkflow):
    ...

# With scope
@register_workflow(scopes=["op"])
class KernelWorkflow(BaseWorkflow):
    ...
```

### API Reference

| Method | Description |
|--------|-------------|
| `WorkflowRegistry.register(cls, name, scopes)` | Register a workflow class. |
| `WorkflowRegistry.get_workflow_class(name)` | Get a workflow class by name. |
| `WorkflowRegistry.list_workflows(scope)` | List registered workflows, optionally filtered by scope. |
| `WorkflowRegistry.get_tool_config(workflow_name)` | Get tool configuration for a workflow (for ToolExecutor dispatch). |
| `WorkflowRegistry.is_registered(name, scope)` | Check if a workflow is registered (in a given scope). |
| `WorkflowRegistry.clear()` | Remove all registered workflows. |

## 7. WorkflowVisualizer

Generate Mermaid diagrams from compiled workflows:

```python
from akg_agents.core_v2.langgraph_base import WorkflowVisualizer

mermaid_str = workflow.visualize()
print(mermaid_str)
```

## 8. Node Tracking

The `track_node` utility helps track node execution within workflows:

```python
from akg_agents.core_v2.langgraph_base import track_node

# Used within workflow nodes to track execution
```
