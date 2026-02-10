# TaskConstructor - Standardized Task Builder Agent

## Overview

TaskConstructor is a self-contained ReAct Agent that extracts operator implementations from existing PyTorch/Triton code and builds them into KernelBench-format standardized single-file tasks.

**Tool Name**: `call_task_constructor`  
**Agent Class**: `TaskConstructor`  
**Module Path**: `akg_agents.op.agents.task_constructor`

### Comparison with OpTaskBuilder

| Aspect | TaskConstructor | OpTaskBuilder |
|--------|----------------|---------------|
| Tool Name | `call_task_constructor` | `call_op_task_builder` |
| Input | Code repo path / file / code snippet | Text-only requirement description |
| Mechanism | Internal ReAct loop + AST analysis + 17 dedicated tools | Single-shot LLM generation + check/retry |
| Use Case | **Has existing code** to extract and standardize | **No code available**, only text description |

## Input / Output

### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_input` | string | Yes | User requirement description |
| `source_path` | string | No | Source code path (file or directory) |

### Output

| Field | Description |
|-------|-------------|
| `status` | `"success"` or `"fail"` |
| `output` | Compact summary (for the calling agent) |
| `task_code` | Complete KernelBench-format code |
| `task_code_path` | Absolute path of the output file |
| `op_name` | Operator name |
| `summary` | Build process summary |

## Workflow

```
User Input → system_prompt + tools → ReAct Loop
                                       │
    ┌──────────────────────────────────┘
    │
    ├─ 1. grep_search: locate target function in source
    ├─ 2. copy_to_workspace: copy source file to workspace
    ├─ 3. read_function: extract target function code
    ├─ 4. trace_dependencies: AST dependency tracing
    ├─ 5. read_function: extract external deps & inline
    ├─ 6. assemble_task: build self-contained task file
    ├─ 7. validate_task → optimize_task → validate_task
    ├─ 8. test_with_reference: multi-input comparison test
    └─ 9. finish: return result
```

## Tool Set (17 tools)

### Code Analysis

| Tool | Description |
|------|-------------|
| `trace_dependencies` | AST dependency tracing, discovers all called functions |
| `assemble_task` | Extract functions from source, assemble self-contained task |
| `validate_task` | Pre-run validation (instantiation → forward → NaN/Inf) |
| `optimize_task` | Clean imports, remove dead code, format |
| `test_with_reference` | Multi-input comparison with original function |

### Code Execution

| Tool | Description |
|------|-------------|
| `run_code` | Run Python code or file |
| `apply_patch` | Modify file via string replacement |

### File Operations

| Tool | Description |
|------|-------------|
| `read_file` | Read file (supports line range) |
| `write_file` | Write file |
| `append_to_file` | Append content |
| `scan_dir` | Scan directory structure |
| `copy_to_workspace` | Copy external file to workspace |
| `read_function` | Precisely extract function/class code |
| `grep_search` | Regex search in files |
| `save_to_workspace` | Save text to workspace file |
| `list_workspace` | List workspace files |
| `multi_file_search` | Multi-file code fragment search |

## Usage

### Option 1: Direct Call (for testing)

```python
import asyncio
from akg_agents.op.agents.task_constructor import TaskConstructor

async def main():
    agent = TaskConstructor()
    result = await agent.run(
        user_input="Extract torch._chunk_cat decomposition, build standardized task",
        source_path="/path/to/pytorch",
    )
    print(result["status"])
    print(result["task_code_path"])

asyncio.run(main())
```

### Option 2: Via KernelAgent Main Flow

KernelAgent auto-registers `call_task_constructor` at startup. The LLM selects it based on the Skill guide:

```python
from akg_agents.op.agents.kernel_agent import KernelAgent

agent = KernelAgent(task_id="test_001")
result = await agent.run(
    user_input="Extract torch._chunk_cat from pytorch repo, generate triton kernel",
    source_path="/path/to/pytorch",
)
```

KernelAgent dispatches to `call_task_constructor` via `ToolExecutor`, with parameters extracted from LLM output.

## Skill System

### Skill File

Located at `op/resources/skills/task-constructor/SKILL.md`

### Resources

| File | Description |
|------|-------------|
| `references/kernelbench-format.md` | KernelBench format specification |
| `references/assembly-strategies.md` | Assembly strategy guide |
| `scripts/validate_kernelbench_task.py` | Format validation script |

### Resource Access

- **references**: Read by LLM via the `read_file` tool
- **scripts**: Executed by LLM via the `execute_script` tool

```
# Read reference document
Action: read_file(file_path="resources/skills/task-constructor/references/kernelbench-format.md")

# Execute validation script
Action: execute_script(
    script_path="resources/skills/task-constructor/scripts/validate_kernelbench_task.py",
    args="--stdin --json",
    stdin_input="<task code>"
)
```

## Logging

Each run generates logs under `~/.akg/task_constructor/<timestamp>/logs/`:

| File | Content |
|------|---------|
| `system_prompt.txt` | Full system prompt |
| `initial_message.txt` | Initial user message |
| `session.log` | Text-format session log |
| `messages.jsonl` | Structured step records |
| `prompt_step_NNN.json` | Complete messages sent to LLM per step |
| `result.json` | Final result |
| `task_output.py` | Generated task code |

## Directory Structure

```
python/akg_agents/op/
├── agents/
│   └── task_constructor.py          # Main agent implementation
├── tools/task_constructor/
│   ├── __init__.py                  # Entry point, triggers tool registration
│   ├── tool_registry.py             # TaskToolRegistry
│   ├── code_tools.py                # Code tool registration entry
│   ├── file_tools.py                # File operation tools
│   ├── path_utils.py                # Shared path resolution
│   ├── ast_utils.py                 # AST parsing
│   ├── code_cleanup.py              # Code cleanup
│   ├── assembly.py                  # Task assembly/validation
│   └── execution.py                 # Code execution/testing
└── resources/
    ├── prompts/task_constructor/
    │   └── system_prompt.j2         # Jinja2 prompt template
    └── skills/task-constructor/
        ├── SKILL.md                 # Skill description
        ├── references/              # Reference docs
        └── scripts/                 # Validation scripts
```
