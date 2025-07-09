# General Designer Design Document

## Overview
Designer is a core component in the AI Kernel Generator that automatically generates and repairs design documents based on Large Language Models (LLMs). It inherits from `AgentBase` and is responsible for intelligently generating high-quality kernel design documents based on the kernel name, task description, and hardware configuration. We currently use AUL (AI Unity Language) as the expression language for design documents, but users can flexibly design other implementation methods.

## Core Functions
- **Intelligent Code Generation**: Automatically generates AUL code based on the kernel name and task description.
- **Automatic Code Repair**: Intelligently fixes code issues based on verification feedback.  
- **Multi-Hardware Support**: Supports hardware backends such as Ascend NPUs and CUDA GPUs.
- **Document Integration**: Automatically loads AUL specifications and hardware documentation.
- **Dynamic Adaptation**: Dynamically obtains configuration information based on the hardware type.

## Initialization Parameters
| Parameter Name | Type/Required | Description |
|---------|---------|---------|
| op_name | str (Required) | Kernel name, e.g., "matmul", "relu" |
| task_desc | str (Required) | Task description, detailing the functional requirements of the kernel. |
| model_config | dict (Required) | LLM model configuration, including configurations for both generation and repair models. |
| impl_type | str (Optional) | Implementation type, e.g., "swft". Default: "" |
| backend | str (Optional) | Hardware backend: cpu/ascend/cuda. Default: "" |
| arch | str (Optional) | Hardware architecture: ascend310p3/ascend910b4/a100. Default: "" |

## Execution Flow run

1. **State Update Stage**
   - Extract existing AUL code (from `parsed_code.aul_code`).
   - Call `update()` to update the agent's state information.

2. **Core Execution Stage**  
   - Select the corresponding processing logic based on `action_type`.
     - `DO_DESIGNER`: Call `run_llm()` using the `aul_gen_prompt` template and `aul_gen_input` data.
     - `FIX_DESIGNER`: Call `run_llm()` using the `aul_fix_prompt` template and `aul_fix_input` data.
   - Throw a `ValueError` for unsupported `action_type`.

3. **Result Return**
   - Returns a triplet: (generated content, formatted prompt, inference content).

## Usage Example
```python
from ai_kernel_generator.core.agent.aul_designer import AULDesigner
from ai_kernel_generator.core.utils import ActionType

# Create a Designer instance
designer = AULDesigner(
    op_name="relu",
    task_desc=task_desc,
    model_config=config["model"],
    impl_type="triton",
    backend="ascend",
    arch="ascend910b4"
)

# Execute code generation
async def generate_code():
    result = await designer.run(
        action_type=ActionType.DO_DESIGNER,
        parsed_code=None,
        suggestions=""
    )
    print(f"Generated AUL code: {result[0]}")

# Execute code repair
async def fix_code():
    result = await designer.run(
        action_type=ActionType.FIX_DESIGNER,
        parsed_code=parsed_code,
        suggestions="Optimize memory access patterns"
    )
    print(f"Fixed code: {result[0]}")
``` 