# Workflow System Design Document

## Overview
AKG Agents uses a **LangGraph**-based workflow system to define and execute task flows, providing better flexibility, type safety, and visualization capabilities.

## Architecture Overview

```
core_v2/langgraph_base/         # Generic base classes (domain-agnostic)
├── base_task.py                # BaseLangGraphTask - task base class
├── base_workflow.py            # BaseWorkflow - workflow base class
└── visualizer.py               # Workflow visualization

op/langgraph_op/                # Operator domain implementation
├── task.py                     # LangGraphTask - operator task executor
├── state.py                    # KernelGenState - operator state definition
├── nodes.py                    # NodeFactory - node factory
└── routers.py                  # RouterFactory - router factory

op/workflows/                   # Operator workflow definitions
├── base_workflow.py            # OpBaseWorkflow - operator workflow base class
├── default_workflow.py         # Standard: Designer → Coder ↔ Verifier
├── coder_only_workflow.py      # Code only: Coder ↔ Verifier
├── kernelgen_only_workflow.py  # Skill System: KernelGen ↔ Verifier
├── verifier_only_workflow.py   # Verify only: Verifier → END
└── connect_all_workflow.py     # Fully connected: All ↔ All
```

## Core Concepts

### BaseLangGraphTask (Generic Task Base Class)
Located in `core_v2/langgraph_base/base_task.py`, provides a generic framework for task execution without any domain-specific logic.

**Core Methods:**
| Method | Description |
|--------|-------------|
| `_init_workflow()` | Abstract method, subclasses implement workflow initialization |
| `_prepare_initial_state()` | Abstract method, subclasses implement initial state preparation |
| `run()` | Execute the task, returns `(success, final_state)` |
| `visualize()` | Generate Mermaid/PNG flowchart |

### BaseWorkflow (Generic Workflow Base Class)
Located in `core_v2/langgraph_base/base_workflow.py`, provides a domain-agnostic workflow abstraction.

**Core Methods:**
| Method | Description |
|--------|-------------|
| `build_graph()` | Abstract method, subclasses implement specific graph structure |
| `compile()` | Compile the workflow graph, returns an executable LangGraph app |
| `visualize()` | Generate Mermaid format flowchart |

### LangGraphTask (Operator Task Executor)
Located in `op/langgraph_op/task.py`, inherits `BaseLangGraphTask`, adding operator generation-specific logic:
- Operator Agent initialization (Designer, Coder, KernelGen, Verifier)
- Device pool and Worker management
- Operator-specific initial state preparation
- Workflow registry (`WORKFLOW_REGISTRY`)

### KernelGenState (State Definition)
Type-safe state defined using `TypedDict`, containing:
- Generic fields: `task_id`, `iteration`, `max_iterations`, `success`, etc.
- Operator-specific fields: `op_name`, `task_desc`, `dsl`, `framework`, `backend`, `arch`, etc.
- Result fields: `verifier_result`, `verifier_error`, `coder_code`, `designer_sketch`, etc.

## Available Workflows

### Workflow Registry
```python
WORKFLOW_REGISTRY = {
    "default":           DefaultWorkflow,       # Standard flow
    "coder_only":        CoderOnlyWorkflow,     # Code only
    "kernelgen_only":    KernelGenOnlyWorkflow, # Skill System
    "verifier_only":     VerifierOnlyWorkflow,  # Verify only
    "connect_all":       ConnectAllWorkflow,    # Fully connected
}
```

### 1. Standard Workflow (`default`)
```
Designer → Coder → Verifier → [Success] → END
                       ↓ [Failure]
                  Conductor → Coder (repeat)
```
Complete design → code → verify flow.

### 2. Code-Only Workflow (`coder_only`)
```
Coder → Verifier → [Success] → END
            ↓ [Failure]
       Conductor → Coder (repeat)
```
Skips the design phase, directly generates code.

### 3. KernelGen Workflow (`kernelgen_only`)
```
KernelGen → Verifier → [Success] → END
                ↓ [Failure]
           Conductor → KernelGen (repeat)
```
Skill System-based code generation with dynamic knowledge injection.

### 4. Verify-Only Workflow (`verifier_only`)
```
Verifier → END
```
Minimal flow, verifies existing code only.

### 5. Fully Connected Workflow (`connect_all`)
Supports flexible transitions between all Agents for maximum flexibility.

## Conductor Node in Workflows

The Conductor node (not a standalone Agent) performs LLM-driven error analysis when verification fails:
1. Loads the `conductor/analyze.j2` template
2. Prepares prompts with error information, historical attempts, etc.
3. Calls LLM for analysis
4. Parses decision (continue/terminate) and suggestions
5. Returns updated state to the next code generation node

## Quick Start

### Basic Usage
```python
from akg_agents.op.langgraph_op.task import LangGraphTask

# Create task
task = LangGraphTask(
    op_name="akg_relu",
    task_desc="Implement ReLU activation function...",
    task_id="0",
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100",
    config=config,
    workflow="default"  # Select workflow
)

# Run task
op_name, success, final_state = await task.run()
```

### Visualization
```python
# Print Mermaid graph
print(task.visualize())

# Save as PNG
task.visualize(output_path="workflow.png")
```

## Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| op_name | str | Yes | Operator name |
| task_desc | str | Yes | Task description (framework code) |
| task_id | str | Yes | Unique task identifier |
| dsl | str | Yes | Target DSL: `triton_cuda`, `triton_ascend`, etc. |
| framework | str | Yes | Frontend framework: `torch`, `mindspore`, `numpy` |
| backend | str | Yes | Backend: `cuda`, `ascend` |
| arch | str | Yes | Hardware architecture: `a100`, `ascend910b4` |
| config | dict | Yes | Configuration dictionary |
| workflow | str | No | Workflow name (default: `default`) |
| inspirations | list | No | Code inspirations from evolution |
| meta_prompts | str | No | Meta prompts for LLM |
| user_requirements | str | No | Additional user requirements |

## Custom Workflows

### Creating a Custom Workflow
```python
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from langgraph.graph import StateGraph, END

class MyWorkflow(OpBaseWorkflow):
    def build_graph(self) -> StateGraph:
        workflow = StateGraph(KernelGenState)

        # Create nodes
        coder_node = NodeFactory.create_coder_node(
            self.agents['coder'], self.trace
        )
        verifier_node = NodeFactory.create_verifier_node(
            self.agents['verifier'], self.device_pool,
            self.trace, self.config
        )

        # Add nodes
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)

        # Define flow
        workflow.set_entry_point("coder")
        workflow.add_edge("coder", "verifier")
        workflow.add_edge("verifier", END)

        return workflow
```

### Registering a Custom Workflow
Add to `WORKFLOW_REGISTRY` in `op/langgraph_op/task.py`:
```python
WORKFLOW_REGISTRY = {
    ...
    "my_workflow": MyWorkflow,
}
```

Or use decorator registration (recommended, supports `KernelAgent` dynamic discovery):
```python
from akg_agents.core_v2.workflows.registry import register_workflow

@register_workflow(scopes=["op"])
class MyWorkflow(OpBaseWorkflow):
    TOOL_NAME = "use_my_workflow"
    DESCRIPTION = "My custom workflow description"
    PARAMETERS_SCHEMA = { ... }
    ...
```

## Workflow as Tool

In the new architecture, workflows can be invoked by `KernelAgent` (ReAct Agent) as tools. Each workflow class can define the following metadata:
- `TOOL_NAME`: Tool name (e.g., `use_kernelgen_only_workflow`)
- `DESCRIPTION`: Tool description (used by LLM for decision-making)
- `PARAMETERS_SCHEMA`: Parameter JSON Schema

`KernelAgent` dynamically discovers and loads workflows registered with `@register_workflow(scopes=["op"])` via `WorkflowRegistry` and exposes them as callable tools.

## Comparison with Old Architecture

| Component | Old Architecture | New Architecture |
|-----------|-----------------|-----------------|
| **Scheduler** | `Conductor` class + `workflow.yaml` | LangGraph `StateGraph` + Python workflows |
| **Workflow Definition** | YAML files | Python classes (`op/workflows/`) |
| **State Management** | `task_info` dict | `KernelGenState` TypedDict |
| **Decision Logic** | Conductor LLM + code logic | Conductor node + Router functions |
| **Visualization** | None | Mermaid / PNG diagrams |
| **Extension** | Edit YAML | Inherit `OpBaseWorkflow` and implement `build_graph()` |
| **Tool Invocation** | None | Workflow as Tool (`KernelAgent` callable) |
