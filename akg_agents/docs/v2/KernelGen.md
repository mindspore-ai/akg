[中文版](./CN/KernelGen.md)

# KernelGen Agent Design Document

## Overview
KernelGen is a Skill System-based kernel code generation Agent in AKG Agents. It inherits from `AgentBase` (`core_v2`) and is responsible for generating high-performance kernel code by dynamically selecting relevant knowledge and strategies through the Skill System based on user input and historical context.

## Core Features
- **Skill-Driven Code Generation**: Two-stage Skill selection (coarse filter + LLM fine filter) for dynamic knowledge loading
- **Multi-DSL Support**: Supports various target DSLs (Triton CUDA, Triton Ascend, AscendC, etc.)
- **Multi-Framework Adaptation**: Supports PyTorch, MindSpore, NumPy and other frontend frameworks
- **Agent Registration**: Automatically registered to the Agent registry via `@register_agent` decorator
- **Tool Configuration**: Provides `TOOL_NAME`, `DESCRIPTION`, `PARAMETERS_SCHEMA` for invocation by KernelAgent as a tool

## Architecture

```
core_v2/agents/base.py          # AgentBase base class
    ↑
op/agents/kernel_gen.py          # KernelGen Agent (inherits AgentBase)
    ↑
op/agents/kernel_agent.py        # KernelAgent (ReAct Agent, calls KernelGen as tool)
```

## Tool Configuration

When KernelGen is called as a tool by KernelAgent:

| Attribute | Value |
|-----------|-------|
| TOOL_NAME | `call_kernel_gen` |
| Use Case | User explicitly says "no verification needed", "just give me the code", "quick draft" |
| Output | Generated kernel code (including class ModelNew and kernel function) |

⚠️ This tool only generates code without verification. For a complete generation + verification flow, use `use_kernelgen_only_workflow`.

## Parameters

| Parameter | Type/Required | Description |
|-----------|--------------|-------------|
| op_name | str (Required) | Operator name |
| task_desc | str (Required) | Task description or algorithm specification |
| dsl | str (Required) | Target DSL: `triton_ascend`, `triton_cuda`, etc. |
| framework | str (Required) | Target framework: `torch`, `mindspore`, `numpy`, etc. |
| backend | str (Required) | Target hardware backend: `cuda`, `ascend`, etc. |
| arch | str (Optional) | Target hardware architecture: `a100`, `ascend910b4`, etc. |
| user_requirements | str (Optional) | Additional user requirements |
| task_id | str (Optional) | Task ID |
| history_compress | list (Optional) | Compressed history record list |
| verifier_error | str (Optional) | Verifier error message |
| conductor_suggestion | str (Optional) | Conductor fix suggestion |
| model_level | str (Optional) | Model level: `standard`, `fast`, `complex`, default `standard` |

## Skill System Integration

### Two-Stage Skill Selection

KernelGen uses a two-stage Skill selection mechanism:

1. **Coarse Filter (Metadata Filtering)**: Uses `OperatorSkillSelector` for quick filtering based on backend/dsl metadata
2. **Fine Filter (LLM Evaluation)**: When candidate Skills exceed 3, LLM selects the most relevant Skills based on task description

```python
# Coarse filter
context = OperatorSelectionContext(dsl="triton-ascend", backend="ascend")
filtered = selector.coarse_filter(loaded_skills, context)

# Fine filter (when candidates > 3)
selected = llm_select(filtered, task_desc, op_name)
```

### Skills Directory
Skills are stored in the `op/resources/skills/` directory. See [Skill System Documentation](./SkillSystem.md) for details.

## Execution Flow

1. **Initialization Stage**
   - Initialize parent class `AgentBase` (configure LLM, load templates, etc.)
   - Create code parser via `parser_loader`
   - Load Skill System (`SkillLoader` loads from skills directory)
   - Load Jinja2 Prompt templates (`system_prompt.j2`, `user_prompt.j2`)

2. **Skill Selection Stage**
   - Execute two-stage Skill selection based on task parameters (op_name, dsl, backend, etc.)
   - Return most relevant Skills list

3. **Prompt Construction Stage**
   - Render system prompt using System Prompt template (with DSL, framework, backend info)
   - Render user prompt using User Prompt template (with history, Skills content, task description, etc.)

4. **Code Generation Stage**
   - Call LLM via `run_llm` to generate code
   - Return `(generated code, full prompt, reasoning process)`

## Usage Examples

### Direct Invocation
```python
from akg_agents.op.agents.kernel_gen import KernelGen

# Initialize
kernel_gen = KernelGen()

# Execute code generation
code, prompt, reasoning = await kernel_gen.run(
    op_name="relu",
    task_desc="Implement ReLU activation function",
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100"
)
```

### Via KernelAgent (as Tool)
KernelGen is registered as the `call_kernel_gen` tool and is automatically invoked by KernelAgent during the ReAct loop based on user needs. See [Workflow Documentation](./Workflow.md) for details.

## Related Documentation
- [KernelDesigner Design Document](./KernelDesigner.md)
- [Workflow System](./Workflow.md)
- [Skill System Documentation](./SkillSystem.md)
- [Trace System Documentation](./Trace.md)
