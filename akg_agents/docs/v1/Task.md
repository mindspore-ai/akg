# Task Module Design Document

## Overview
The Task module is a core component of the AI Kernel Generator, responsible for executing the design, coding, verification, and optimization flow for a single kernel. It integrates four core components: Designer, Coder, Verifier, and Conductor, implementing the complete conversion and verification process from AUL code to Triton/SWFT code.


## Core Functions
- **Task Lifecycle Management**: Manages the complete execution flow from initialization to verification completion.
- **Multi-Component Coordination**: Integrates the four core components: Designer, Coder, Verifier, and Conductor.
- **Hardware Resource Scheduling**: Manages the allocation and release of Ascend/NVIDIA devices through DevicePool.
- **Execution Control**: Controls the maximum number of iteration steps through the `limit_steps` parameter.


## Initialization Parameters
| Parameter Name | Type/Required | Description |
|---------|---------|---------|
| op_name | str (Required) | Kernel name (e.g., "matmul") |
| task_desc | str (Required) | Task description, supports only MindSpore/Torch/NumPy implementations. |
| task_id | str (Required) | Task ID, used to distinguish different shapes or computations when `op_name` is the same. |
| backend | str (Required) | Computation backend, supports only ascend/cuda/cpu |
| arch | str (Required) | Hardware architecture, which varies depending on the backend, e.g., ascend910b4/a100. |
| impl_type | str (Required) | Backend implementation type, only triton and swft |
| config | dict (Required) | Contains LLM configuration, log paths, etc. |
| device_pool | DevicePool (Required) | The device resource pool. |
| framework | str (Required) | Framework type, supports only mindspore/torch/numpy |
| task_type | str (Optional) | Task type, supports `precision_only` (verifies result correctness) or `profile` (for performance analysis). Default: `precision_only` |
| limit_steps | int (Optional) | Maximum number of execution steps. Default: 10 |


## Execution Flow run

1. **Initialization Stage**
   - Initialize the Conductor control module.
   - Load Designer configuration parameters.
   - Configure Coder code templates.
   - Prepare the Verifier environment.

2. **Core Execution Stage**
   - Determine the execution step based on the `action_type` decided by the Conductor.
     - `designer`: Call Designer to generate AUL code.
     - `coder`: Call Coder to convert to Triton/SWFT code.
     - `verifier`: Call Verifier for precision/performance verification and release device resources.

3. **Logging and Iteration**
   - Add logs to the Conductor's log queue.
   - Loop execution until the maximum number of iterations is reached or verification passes.

## Usage Example
```python
# Create a task instance
task = Task(
    op_name="swish",
    task_desc="Swish activation function: x * sigmoid(beta * x)",
    backend="ascend",
    arch="ascend310p3",
    impl_type="swft",
    config=load_config(),
    device_pool=global_device_pool
)

# Execute the task
async def run_task():
    success = await task.run()
    print(f"Task completed: {success}")
``` 