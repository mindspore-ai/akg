# AKG Agents Example Tutorial

## Overview

This tutorial demonstrates how to use the AKG Agents LangGraph task system to generate high-performance kernel code. Based on `LangGraphTask` (new architecture), it shows the complete code generation and verification flow.

## Core Components

### Main Modules
- **LangGraphTask**: Task executor based on the LangGraph workflow engine
- **TaskPool**: Task pool manager for async batch task execution
- **WorkerManager**: Worker service management for remote/local device scheduling

### Execution Flow
```
Task Initialization → Workflow Selection → Agent Execution (Design/Code/Verify) → Result Output
```

## Example 1: PyTorch Triton Single Task (CUDA)

See `examples/run_torch_triton_single.py`.

### 1. Task Description Function

```python
def get_task_desc():
    return '''
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]


def get_init_inputs():
    return []
'''
```

**Key Elements**:
- **Model Definition**: PyTorch model class inheriting `nn.Module`
- **Operator Implementation**: Define operator logic in the `forward` method
- **Input Generation**: `get_inputs()` generates test data (specify device)
- **Initialization Inputs**: `get_init_inputs()` provides model initialization parameters

### 2. Main Execution Function

```python
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.core.task_pool import TaskPool
from akg_agents.config.config_validator import load_config, check_env_for_task

async def run_torch_triton_single():
    op_name = "akg_relu"
    task_desc = get_task_desc()

    task_pool = TaskPool()

    # Register local Worker (specify GPU device list)
    await register_local_worker([0], backend="cuda", arch="a100")

    # Load configuration
    config = load_config(dsl="triton_cuda", backend="cuda")

    # Recommended: environment check before running
    check_env_for_task("torch", "cuda", "triton_cuda", config)

    # Create LangGraph task
    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton_cuda",
        backend="cuda",
        arch="a100",
        config=config,
        framework="torch",
        workflow="coder_only_workflow"  # Workflow selection
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()

    for op_name, success, task_info in results:
        if success:
            print(f"Task {op_name} passed")
        else:
            print(f"Task {op_name} failed")
```

### Parameter Descriptions

| Parameter | Type | Description |
|-----------|------|-------------|
| op_name | str | Operator name |
| task_desc | str | Task description (model definition and test data generation) |
| task_id | str | Unique task identifier |
| dsl | str | Target DSL: `triton_cuda`, `triton_ascend`, etc. |
| backend | str | Compute backend: `cuda`, `ascend`, `cpu` |
| arch | str | Hardware architecture: `a100`, `ascend910b4`, etc. |
| config | dict | Task configuration (loaded via `load_config`) |
| framework | str | Frontend framework: `torch`, `mindspore`, `numpy` |
| workflow | str | Workflow name (default: `default`), options: `coder_only`, `kernelgen_only`, `verifier_only` |

## Example 2: Ascend NPU Triton Single Task

See `examples/run_torch_npu_triton_single.py`.

```python
async def run_torch_npu_triton_single():
    op_name = "akg_relu"
    task_desc = get_task_desc()  # Task description with npu tensors

    task_pool = TaskPool()

    # Register Ascend device
    await register_local_worker([0], backend="ascend", arch="ascend910b4")

    config = load_config(dsl="triton_ascend", backend="ascend")
    check_env_for_task("torch", "ascend", "triton_ascend", config)

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        config=config,
        framework="torch",
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
```

## Example 3: CPU C++ Single Task

See `examples/run_torch_cpu_cpp_single.py`.

```python
async def run_torch_cpu_cpp_single():
    op_name = "akg_relu"
    task_desc = get_task_desc()  # CPU version task description

    task_pool = TaskPool()

    # Register CPU Worker
    await register_local_worker([0], backend="cpu", arch="x86_64")

    config = load_config("cpp")
    check_env_for_task("torch", "cpu", "cpp", config)

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        config=config,
        framework="torch",
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
```

## Available Workflows

| Workflow | Flow | Use Case |
|----------|------|----------|
| `default` | Designer → Coder ↔ Verifier | Full flow: design → code → verify |
| `coder_only` | Coder ↔ Verifier | Skip design, directly generate code |
| `kernelgen_only` | KernelGen ↔ Verifier | Skill System-based code generation |
| `verifier_only` | Verifier → END | Verify existing code only |
| `connect_all` | All ↔ All | Fully connected, maximum flexibility |

## Workflow Visualization

```python
# After creating a task, visualize the workflow
task = LangGraphTask(...)

# Print Mermaid flowchart
print(task.visualize())

# Save as PNG file
task.visualize(output_path="workflow.png")
```

## Running Steps

### 1. Environment Setup

See the project's [README](../README.md).

### 2. Configure LLM

Configure LLM via environment variables or `settings.json`:
```bash
# Environment variable method
export AKG_AGENTS_LLM_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_LLM_API_KEY="sk-xxx"
export AKG_AGENTS_LLM_MODEL_NAME="gpt-4"
```

Or configure in `.akg/settings.json`:
```json
{
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxx",
    "model_name": "gpt-4"
  }
}
```

### 3. Execute Examples

```bash
# CUDA A100
python examples/run_torch_triton_single.py

# Ascend 910B
python examples/run_torch_npu_triton_single.py

# CPU C++
python examples/run_torch_cpu_cpp_single.py
```

## Related Documentation
- [Workflow System Documentation](./Workflow.md)
- [KernelGen Agent Documentation](./KernelGen.md)
- [KernelDesigner Agent Documentation](./KernelDesigner.md)
- [AKG CLI Documentation](./AKG_CLI.md)
