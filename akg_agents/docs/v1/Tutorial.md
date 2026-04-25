# AI Kernel Generator Example Tutorial (Coder-only)

## Overview

This tutorial is based on the coder-only workflow. It demonstrates how to generate a Triton implementation directly from a MindSpore frontend description and verify it (without the Designer step). See `examples/run_mindspore_triton_single.py`.

## Core Components

### Main Modules
- **Task**: A task instance
- **TaskPool**: A task pool manager for async execution
- **DevicePool**: A device resource pool for device allocation

### Execution Flow (coder-only)
```
Task Initialization → Coder generates Triton code → Verifier verifies the result
```

## Example Code

### 1. Task Description Function

```python
def get_task_desc():
    return '''
import mindspore as ms
from mindspore import nn


class Model(nn.Cell):
    def __init__(self):
        super(Model, self).__init__()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return ms.ops.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = ms.ops.randn(batch_size, dim, dtype=ms.float16)
    return [x]


def get_init_inputs():
    return []
'''
```

**Key Elements**:
- Model definition using `nn.Cell`
- Kernel logic defined in `construct`
- `get_inputs()` for test data
- `get_init_inputs()` for init params

### 2. Main Execution Function (coder-only)

```python
async def run_mindspore_triton_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    device_pool = DevicePool([0])
    config = load_config(dsl="triton_ascend", backend="ascend")  # choose default plan by DSL

    # Recommended: environment check before running
    check_env_for_task("mindspore", "ascend", "triton_ascend", config)

    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        config=config,
        device_pool=device_pool,
        framework="mindspore",
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    # Process results...
```

**Parameter Descriptions**:
| Name | Type | Description |
|---------|------|------|
| op_name | str | Kernel name |
| task_desc | str | Task description (model + inputs) |
| task_id | str | Unique task identifier |
| dsl | str | Target DSL, e.g. "triton_cuda", "triton_ascend", "swft" |
| backend | str | Backend, e.g. "ascend", "cuda" |
| arch | str | Hardware arch, e.g. "ascend910b4" |
| config | dict | Task orchestration plan config (`agent_model_config`, `workflow_config_path`, `docs_dir`, etc.) |
| device_pool | DevicePool | Device pool |
| framework | str | Frontend framework: "mindspore"/"torch"/"numpy" |
| workflow | str | Optional. Override `workflow_config_path`, e.g. "coder_only_workflow" |

> Configuration: `load_config("triton_ascend", backend="ascend")` loads `config/default_triton_ascend_config.yaml` for Ascend backend, or `load_config("triton_cuda", backend="cuda")` for CUDA backend. If you run with local vLLM and coder-only, use `vllm_triton_ascend_coderonly_config.yaml` (Ascend) or `vllm_triton_cuda_coderonly_config.yaml` (CUDA) via `load_config(config_path=...)`.

## Run

### 1. Environment Setup

See the project's [README](../README.md).

### 2. Execute the example

```bash
python examples/run_mindspore_triton_single.py
```