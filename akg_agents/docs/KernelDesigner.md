# KernelDesigner Agent Design Document

## Overview
KernelDesigner is a Skill System-based algorithm sketch design Agent in AKG Agents. It inherits from `AgentBase` (`core_v2`) and is responsible for generating algorithm sketches that provide implementation plans and optimization strategies based on user input and historical context.

## Core Features
- **Intelligent Algorithm Sketch Design**: Generates high-quality algorithm sketches (pseudocode + optimization strategies + implementation suggestions) based on task requirements
- **Skill-Driven Knowledge Selection**: Automatically selects mandatory sketch-design Skill and uses LLM fine-filtering for reference Skills
- **Hint Mode**: Supports parameter space configuration; automatically enabled when task description contains hint information
- **Multi-DSL Support**: Supports Triton CUDA, Triton Ascend, and other DSLs
- **Hardware-Aware Design**: Automatically loads hardware documentation, considering hardware characteristics during design
- **Agent Registration**: Automatically registered to the Agent registry via `@register_agent` decorator

## Architecture

```
core_v2/agents/base.py            # AgentBase base class
    ↑
op/agents/kernel_designer.py       # KernelDesigner Agent (inherits AgentBase)
    ↑
op/agents/kernel_agent.py          # KernelAgent (ReAct Agent, calls KernelDesigner as tool)
```

## Tool Configuration

When KernelDesigner is called as a tool by KernelAgent:

| Attribute | Value |
|-----------|-------|
| TOOL_NAME | `call_kernel_designer` |
| Use Case | User says "design plan", "algorithm sketch", "sketch", "how to implement", needs to discuss design first |
| Output | Algorithm sketch (pseudocode + optimization strategies + implementation suggestions) |

⚠️ This tool only generates design plans, not executable code. For design + code generation, use `use_default_workflow`.

## Parameters

| Parameter | Type/Required | Description |
|-----------|--------------|-------------|
| op_name | str (Required) | Operator name |
| task_desc | str (Required) | Task description or algorithm specification |
| dsl | str (Required) | Target DSL: `triton_ascend`, `triton_cuda`, etc. |
| backend | str (Required) | Target hardware backend: `cuda`, `ascend`, etc. |
| arch | str (Optional) | Target hardware architecture: `a100`, `ascend910b4`, etc. |
| task_id | str (Optional) | Task ID |
| user_requirements | str (Optional) | Additional user requirements |
| enable_hint_mode | bool (Optional) | Whether to enable Hint mode (parameter space configuration), default False |
| history_compress | list (Optional) | Compressed history record list |
| model_level | str (Optional) | Model level: `standard`, `fast`, `complex`, default `standard` |

## Skill System Integration

### Designer's Skill Selection Strategy

KernelDesigner uses a different Skill selection strategy from KernelGen:

1. **Mandatory Skill**: `sketch-design` (always loaded)
2. **Conditional Skill**: `hint-mode` (loaded when `enable_hint_mode=True` and hint information is detected)
3. **LLM Fine Filter**: From coarse-filtered candidates, LLM selects 2-3 most relevant reference Skills

```python
# Designer custom filter
def designer_filter(skill, context):
    # Prefer design / fundamental categories
    # Exclude pure implementation categories (used by KernelGen)
    design_categories = ["design", "fundamental"]
    is_designer_skill = skill.category in design_categories
    implementation_only = skill.category == "implementation" and not is_designer_skill
    return not implementation_only
```

### Hardware Document Loading

KernelDesigner automatically loads corresponding hardware documentation based on backend and arch parameters:

```python
from akg_agents.utils.hardware_utils import get_hardware_doc
hardware_docs = get_hardware_doc(backend, arch)
```

## Execution Flow

1. **Initialization Stage**
   - Initialize parent class `AgentBase`
   - Create designer parser via `parser_loader`
   - Load Skill System (with custom designer filter)
   - Load Jinja2 Prompt templates (`system_prompt.j2`, `user_prompt.j2`)

2. **Skill Selection Stage**
   - Load mandatory Skill (`sketch-design`)
   - Conditionally load `hint-mode` Skill
   - Apply custom filter for coarse filtering
   - LLM fine-filters 2-3 reference Skills from candidates

3. **Prompt Construction Stage**
   - Render System Prompt (with DSL, backend, architecture info)
   - Load hardware documentation
   - Render User Prompt (with history, Skills, task description, hardware docs, etc.)

4. **Design Generation Stage**
   - Call LLM via `run_llm` to generate algorithm sketch
   - Return `(generated sketch, full prompt, reasoning process)`

## Usage Examples

### Direct Invocation
```python
from akg_agents.op.agents.kernel_designer import KernelDesigner

# Initialize
designer = KernelDesigner()

# Execute design
sketch, prompt, reasoning = await designer.run(
    op_name="softmax",
    task_desc="Implement fused softmax operator",
    dsl="triton_cuda",
    backend="cuda",
    arch="a100"
)
```

### Via KernelAgent (as Tool)
KernelDesigner is registered as the `call_kernel_designer` tool and is automatically invoked by KernelAgent during the ReAct loop based on user needs. See [Workflow Documentation](./Workflow.md) for details.

## Related Documentation
- [KernelGen Agent Design Document](./KernelGen.md)
- [Workflow System](./Workflow.md)
- [Skill System Documentation](./SkillSystem.md)
- [Trace System Documentation](./Trace.md)
