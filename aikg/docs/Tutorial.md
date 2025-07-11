# AI Kernel Generator Example Tutorial

## Overview

This tutorial introduces how to use the AI Kernel Generator to run the generation and verification process for a single kernel. Using `examples/run_mindspore_triton_single.py` as an example, it demonstrates how to automatically generate a Triton implementation from a MindSpore kernel definition and verify it.

## Core Components

### Main Modules
- **Task**: A task instance.
- **TaskPool**: A task pool manager that supports asynchronous task execution.
- **DevicePool**: A device resource pool that manages device allocation for execution.

### Execution Flow
```
Task Initialization → AULDesigner generates AUL code → TritonCoder converts to Triton code → Verifier verifies the result
```

## Example Code Analysis

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
- **Model Definition**: A MindSpore model class that inherits from `nn.Cell`.
- **Kernel Implementation**: The specific kernel logic is defined in the `construct` method.
- **Input Generation**: The `get_inputs()` function generates test data.
- **Initialization Inputs**: The `get_init_inputs()` function provides model initialization parameters.

### 2. Main Execution Function

```python
async def run_mindspore_triton_single():
    op_name = get_op_name()
    task_desc = get_task_desc()

    task_pool = TaskPool()
    device_pool = DevicePool([0])
    config = load_config()

    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        impl_type="triton",
        backend="ascend",
        arch="ascend910b4",
        config=config,
        device_pool=device_pool,
        framework="mindspore"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    # Process results...
```

**Parameter Descriptions**:
| Parameter Name | Type | Description |
|---------|------|------|
| op_name | str | Kernel name. |
| task_desc | str | Task description (includes model definition and test data generation). |
| task_id | str | Unique identifier for the task. |
| impl_type | str | Implementation type: "triton" or "swft". |
| backend | str | Computation backend: "ascend", "cuda", etc. |
| arch | str | Hardware architecture: e.g., "ascend910b4". |
| config | dict | Configuration file, including LLM model configurations, etc. |
| device_pool | DevicePool | Device resource pool. |
| framework | str | Framework type: "mindspore", "torch", or "numpy". |

## Running the Steps

### 1. Environment Setup

Please refer to the project's [README documentation](../README.md).

### 2. Running the Example

```bash
python examples/run_mindspore_triton_single.py
``` 