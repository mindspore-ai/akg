# LangGraph Task Design Document

## Overview
LangGraph Task is the new intelligent task scheduler that replaces the original `Conductor + workflow.yaml` system. It uses the LangGraph framework to define workflows in Python code, providing better flexibility, visualization, and type safety while maintaining full backward compatibility.

## Core Features
- **Python-defined Workflows**: Define execution flows in code instead of YAML configurations
- **Graph Visualization**: Generate Mermaid diagrams and PNG images of workflows
- **Type-safe State**: Use `TypedDict` for better type hints and IDE support
- **Backward Compatible**: Drop-in replacement for original `Task` class with identical API

## Quick Start

### Basic Usage
```python
from ai_kernel_generator.core.langgraph_task import LangGraphTask
from ai_kernel_generator.core.worker.manager import register_local_worker

# Register worker
await register_local_worker([0], backend='cuda', arch='a100')

# Create task (same API as original Task)
task = LangGraphTask(
    op_name="aikg_relu",
    task_desc="implement ReLU activation",
    task_id="0",
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100",
    config=config,
    device_pool=None,  # Use WorkerManager instead
    workflow="default_workflow"
)

# Run task
op_name, success, final_state = await task.run()
```

### Visualization
```python
# Print Mermaid diagram
print(task.visualize())

# Save as PNG
task.visualize(output_path="workflow.png")
```

## Architecture Comparison

| Component | Original (Task) | New (LangGraphTask) |
|-----------|----------------|---------------------|
| **Scheduler** | `Conductor` class + `workflow.yaml` | LangGraph `StateGraph` + Python workflows |
| **Workflow Definition** | YAML files (`workflow_config_path`) | Python classes (`workflows/` directory) |
| **Agent Configuration** | Mixed in `workflow.yaml` | Separate `parser_config.yaml` |
| **State Management** | `task_info` dict | `KernelGenState` TypedDict |
| **Decision Logic** | Conductor LLM + code logic | Conductor node + Router functions |
| **Visualization** | None | Mermaid / PNG diagrams |

## Available Workflows

| Workflow Name | Flow | Description |
|--------------|------|-------------|
| `default_workflow` | Designer → Coder ↔ Verifier ↔ Conductor | Full design→code→verify flow |
| `coder_only_workflow` | Coder ↔ Verifier ↔ Conductor | Skip design, generate code directly |
| `verifier_only_workflow` | Verifier → Finish | Verify existing code only |
| `connect_all_workflow` | All ↔ All | Fully connected agents |

## Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| op_name | str | Yes | Kernel name |
| task_desc | str | Yes | Task description (framework code) |
| task_id | str | Yes | Unique task identifier |
| dsl | str | Yes | Target DSL: "triton_cuda", "triton_ascend", "swft" |
| framework | str | Yes | Frontend framework: "torch", "mindspore", "numpy" |
| backend | str | Yes | Backend: "cuda", "ascend" |
| arch | str | Yes | Hardware arch: "a100", "ascend910b4" |
| config | dict | Yes | Configuration dictionary |
| device_pool | DevicePool | No | Device pool (deprecated, use WorkerManager) |
| workflow | str | No | Workflow name (default: "default_workflow") |
| inspirations | list | No | Code inspirations from evolution |
| meta_prompts | str | No | Meta prompts for LLM |
| handwrite_suggestions | list | No | Handwritten optimization suggestions |

## Configuration Changes

### Deprecated Configuration
```yaml
# No longer needed (LangGraphTask ignores this)
workflow_config_path: "config/default_workflow.yaml"
```

### New Configuration
```yaml
# Optional, defaults to config/parser_config.yaml
# parser_config_path: "config/parser_config.yaml"

# Workflow and limits (priority: Task parameter > config > default)
default_workflow: "default_workflow"  # Default workflow name
max_step: 20  # Maximum iteration steps
```

## Workflow Execution Flow

### Default Workflow
```
designer → coder → verifier → [success] → finish
                        ↓ [fail]
                   conductor → coder (repeat)
```

### Coder-Only Workflow
```
coder → verifier → [success] → finish
            ↓ [fail]
       conductor → coder (repeat)
```

## Migration Guide

### Step 1: Replace Import
```python
# Before
from ai_kernel_generator.core.task import Task

# After
from ai_kernel_generator.core.langgraph_task import LangGraphTask
```

### Step 2: Replace Instantiation
```python
# Before
task = Task(...)

# After
task = LangGraphTask(...)  # API is identical
```

### Step 3: Update Worker Registration (Optional but Recommended)
```python
# Before (deprecated but still works)
device_pool = DevicePool([0, 1, 2, 3])
task = LangGraphTask(..., device_pool=device_pool)

# After (new service-oriented approach)
await register_local_worker([0, 1, 2, 3], backend='cuda', arch='a100')
task = LangGraphTask(..., device_pool=None)
```

## File Mapping

### New Files
| File | Description |
|------|-------------|
| `core/langgraph_task.py` | New Task class replacing `task.py` |
| `utils/langgraph/state.py` | State definition (`KernelGenState`) |
| `utils/langgraph/nodes.py` | Node factory (wraps Agents as nodes, includes Conductor node) |
| `utils/langgraph/routers.py` | Router functions (conditional edge logic) |
| `workflows/base_workflow.py` | Workflow base class |
| `workflows/default_workflow.py` | Default workflow |
| `workflows/coder_only_workflow.py` | Coder-only workflow |
| `workflows/verifier_only_workflow.py` | Verifier-only workflow |
| `workflows/connect_all_workflow.py` | Fully connected workflow |
| `utils/parser_loader.py` | Parser loader |
| `config/parser_config.yaml` | Agent parser configuration |

### Modified Files
| File | Changes |
|------|---------|
| `core/evolve.py` | Import and use `LangGraphTask` |
| `utils/evolve/evolution_processors.py` | Import and use `LangGraphTask` |
| `core/agent/designer.py` | Support `parser_config_path`, Hint mode output conversion |
| `core/agent/coder.py` | Support `parser_config_path` |
| `config/*.yaml` | Remove `workflow_config_path`, add `default_workflow` and `max_step` |

### Preserved Files (Backward Compatibility)
| File | Status | Note |
|------|--------|------|
| `core/task.py` | Preserved | Original Task class, some tests still use it |
| `core/agent/conductor.py` | Preserved | Original Conductor class, used by old Task |
| `config/*_workflow.yaml` | Preserved | Used by old Task, LangGraphTask ignores them |
| `utils/workflow_manager.py` | Preserved | Used by old Task |
| `utils/workflow_controller.py` | Preserved | Used by old Conductor only |

## Conductor Analysis

### Trigger Conditions
- **Verifier Fails**: Execute Conductor analysis
- **Verifier Succeeds**: Skip Conductor, finish directly

### Conductor Node
The Conductor node performs LLM-based error analysis:
1. Load `conductor/analyze.j2` template
2. Prepare prompt with error information
3. Call LLM for analysis
4. Parse decision and suggestions
5. Save to trace and files
6. Return updated state

### File Outputs
- Decision and suggestion saved to `{log_dir}/{op_name}/I{task_id}_S{step:02d}_conductor/`
- Same format as original Conductor implementation

## Custom Workflows

### Create Custom Workflow
```python
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from langgraph.graph import StateGraph, START, END

class MyWorkflow(BaseWorkflow):
    def build_graph(self):
        workflow = StateGraph(KernelGenState)
        
        # Create nodes
        designer_node = NodeFactory.create_designer_node(
            self.agents['designer'], self.trace, self.config
        )
        coder_node = NodeFactory.create_coder_node(
            self.agents['coder'], self.trace
        )
        
        # Add nodes
        workflow.add_node("designer", designer_node)
        workflow.add_node("coder", coder_node)
        
        # Define flow
        workflow.add_edge(START, "designer")
        workflow.add_edge("designer", "coder")
        workflow.add_edge("coder", END)
        
        return workflow.compile()
```

### Register Workflow
Add to `langgraph_task.py`:
```python
WORKFLOW_REGISTRY = {
    "default": DefaultWorkflow,
    "my_workflow": MyWorkflow,  # Add here
    # ...
}
```