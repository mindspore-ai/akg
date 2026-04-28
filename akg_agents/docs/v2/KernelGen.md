[中文版](./CN/KernelGen.md)

# KernelGen Agent Design Document

## Overview
KernelGen is a Skill System-based kernel code generation Agent in AKG Agents. It inherits from `AgentBase` (`core_v2`) and is responsible for generating high-performance kernel code by dynamically selecting relevant knowledge and strategies through the Skill System based on user input and historical context.

## Core Features
- **Layered Stage-Based Skill Selection**: Three-layer skill selection (L0 always-inject, L1 LLM-selected guides+examples, L2 LLM-selected cases) adapted per generation stage (initial/debug/optimize)
- **Backend Coarse Filter**: Pre-filters all skills by backend metadata before stage-based selection
- **AB Test Support**: `exclude_skill_names` / `force_skill_names` for precise A/B testing of evolved skills
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
| extra_skills | list (Optional) | Extra skill objects injected after selection, bypassing all filters |
| exclude_skill_names | list[str] (Optional) | Skill names to exclude (AB test A mode) |
| force_skill_names | list[str] (Optional) | Skill names to force-include (AB test B mode) |

## Skill System Integration

### Layered Stage-Based Skill Selection

KernelGen uses a layered stage-based skill selection mechanism adapted to the current generation stage (`initial`, `debug`, `optimize`):

**Pre-filter**: Backend coarse filter via `OperatorSkillSelector.coarse_filter()` removes skills incompatible with the target backend.

**Layer 0 (Always Inject)**: `fundamental` and `reference` category skills are always included regardless of stage.

**Layer 1 (LLM-Selected Guides + Examples)**: `guide` category skills are selected by LLM based on task description and operator characteristics. `example` skills are matched by the `operator_type` of the selected guides.

**Layer 2 (LLM-Selected Cases)**: `case` category skills are only included in `debug` (fix cases) and `optimize` (improvement cases) stages. Cases are selected in the same LLM call as guides.

**Stage Mapping**:

| Stage | Trigger | Included Categories |
|-------|---------|-------------------|
| `initial` | First generation | fundamental, reference, guide, example |
| `debug` | `verifier_error` present | fundamental, reference, guide, example, case (fix) |
| `optimize` | `inspirations` present | fundamental, reference, guide, example, case (improvement) |

**AB Test Control**:
- `exclude_skill_names`: Skills matching these names are removed before selection (A mode — baseline without evolved skills)
- `force_skill_names`: Skills matching these names are force-appended after LLM selection (B mode — ensure evolved skills are included)

These can be set as instance attributes or passed as `run()` parameters (which temporarily override instance attributes).

```python
kernel_gen = KernelGen()

# AB test A mode: exclude evolved skills
kernel_gen.exclude_skill_names = ["triton-ascend-error-fix", "triton-ascend-case-reduce-opt"]

# AB test B mode: force-include evolved skills
kernel_gen.force_skill_names = ["triton-ascend-error-fix"]

# Extra skills: bypass all filters, appended after selection
kernel_gen.extra_skills = [my_custom_skill]
```

### Skills Directory
Skills are stored in the `op/resources/skills/` directory. Evolved skills can be symlinked from `~/.akg/evolved_skills/{dsl}/` into the standard skills directory for automatic discovery. See [Skill System Documentation](./SkillSystem.md) for details.

## Execution Flow

1. **Initialization Stage**
   - Initialize parent class `AgentBase` (configure LLM, load templates, etc.)
   - Create code parser via `parser_loader`
   - Load Skill System (`SkillLoader` loads from skills directory)
   - Load Jinja2 Prompt templates (`system_prompt.j2`, `user_prompt.j2`)

2. **Skill Selection Stage**
   - Apply backend coarse filter, then layered stage-based selection (L0 always-inject, L1 LLM guide+example, L2 LLM case)
   - Apply `exclude_skill_names` before selection and `force_skill_names` after selection (for AB test)
   - Append `extra_skills` (if any) after all selection, ensuring specified Skills are always included
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
