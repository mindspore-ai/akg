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
User Input -> system_prompt + tools -> ReAct Loop
                                         |
    +------------------------------------+
    |
    +-- 1. grep_search: locate target function in source
    +-- 2. copy_to_workspace: copy source file to workspace
    +-- 3. read_function: extract target function code
    +-- 4. trace_dependencies: AST dependency tracing
    +-- 5. read_function: extract external deps and inline
    +-- 6. assemble_task: build self-contained task file
    +-- 7. validate_task -> optimize_task -> validate_task
    +-- 8. test_with_reference: multi-input comparison test
    +-- 9. finish: return result
```

## Tool Set in Detail (17 tools)

### Code Analysis and Assembly Tools

#### `trace_dependencies`

AST dependency tracing tool. Given a source file and entry function names, automatically discovers all same-file functions called directly or indirectly through AST parsing. Also identifies external module calls via import aliases and annotates their source module paths. Returns the full dependency chain, suggested functions to embed, and external dependencies that need manual handling.

- **Parameters**: `file_path` (workspace file path), `entry_functions` (list of entry function names)
- **Implementation**: Based on `trace_function_deps()` in `ast_utils.py`, resolves import alias mappings, recursively traces call chains

#### `assemble_task`

Selective assembly tool. Extracts specified functions + Model class + get_inputs from workspace source files to build a self-contained task file. Supports three modes: full embed (entire file), selective extract (specified function list), and exclusion-based embed (exclude unwanted functions). Uses AST parsing for precise extraction, auto-includes file header imports, auto-removes decorators and unused imports.

- **Parameters**: `source_files` (source file list), `model_code`, `get_inputs_code`, `get_init_inputs_code`, `imports_code` (optional), `helper_code` (optional)
- **Implementation**: Calls `extract_functions()` from `ast_utils.py` for precise extraction, `cleanup_task_code()` from `code_cleanup.py` for basic cleanup

#### `validate_task`

Pre-run validation tool. Performs four-step verification on task code: 1) Instantiate Model class; 2) Call forward for execution; 3) Check output for NaN/Inf; 4) Run multiple times for consistency. Executes in an isolated subprocess with automatic timeout.

- **Parameters**: `task_code` (code string) or `task_file` (file path), `timeout` (default 60s)
- **Implementation**: Generates validation script in temp file, executes via `subprocess.run()`, parses stdout/stderr

#### `optimize_task`

Code optimization and cleanup tool. Called after validate passes, before finish. Performs: dedup imports (merge duplicate `from X import a, b`), remove private module references (internal `_`-prefixed module imports), clean unused code, format. Re-validates after optimization to ensure no breakage.

- **Parameters**: `task_file` (file path) or `task_code` (code string)
- **Implementation**: Calls `polish_task_code()` and `merge_from_imports_text()` from `code_cleanup.py`

#### `test_with_reference`

Correctness comparison tool. Compares Model output against a reference function with multiple input sets. User provides `reference_code` (defining `reference_forward(inputs, init_inputs)`) and optional `multi_inputs_code` (defining `get_multi_test_inputs()` returning test cases). Automatically saves reference and multi_inputs code to output directory for reproducibility.

- **Parameters**: `reference_code` (required), `task_file` or `task_code`, `multi_inputs_code` (optional), `timeout`
- **Implementation**: Generates comparison test script, executes via subprocess, compares outputs per case (allclose)

### Code Execution Tools

#### `run_code`

Python code execution tool. Accepts `code` (code string) or `file_path` (Python file), executes in an isolated subprocess. Captures stdout/stderr. Uses workspace directory as working directory.

- **Parameters**: `code` or `file_path` (at least one), `timeout` (default 30s)

#### `apply_patch`

File modification tool. Finds `old_string` and replaces with `new_string`. If `old_string` is empty, creates new file with `new_string`. Supports workspace path prefix auto-resolution.

- **Parameters**: `file_path`, `old_string`, `new_string`

### File Operation Tools

#### `read_file`

File reader with optional line range (`start_line`, `end_line`). Auto-truncates files exceeding 300 lines with a hint to use line ranges. Supports workspace and external paths.

#### `write_file`

Writes content to specified file, auto-creates parent directories. Supports workspace path prefix.

#### `append_to_file`

Appends content to end of file (auto-adds newline separator). Suitable for incremental large file construction.

#### `scan_dir`

Directory scanner. Lists files with sizes, supports depth limiting (`max_depth`, default 3) and file count limiting (`max_files`, default 100). Shows line counts for `.py` files.

#### `copy_to_workspace`

Copies an external file into the workspace directory. Skips if already exists. First step in the code extraction pipeline.

#### `read_function`

Precise function extractor using AST parsing. Extracts complete function/class definitions (including decorators) from a file. Supports regex matching and multi-function extraction. More efficient than `read_file` + manual search for large files.

- **Parameters**: `file_path`, `function_name` (supports regex, e.g. `_chunk_cat|_pad_chunk`)

#### `grep_search`

Regex search tool. Searches for regex matches under a given path, returns matching lines with context. Supports recursive directory search, auto-ignores binary files and common non-code directories. Not restricted to workspace for external repo searches.

- **Parameters**: `pattern` (regex), `path` (search path), `max_results` (default 20)

#### `save_to_workspace`

Saves text content directly to a named file in workspace. Simpler than `write_file` -- no path construction needed.

#### `list_workspace`

Lists all files in the current workspace directory with sizes.

#### `multi_file_search`

Multi-file code fragment search. Searches for a pattern across multiple files simultaneously, returns matches per file.

## Tool Dependency Graph

```
Phase1_Search          Phase2_Extract              Phase3_Build         Phase4_Validate
+-----------+        +----------------+          +--------------+    +---------------+
|grep_search|---+--->| read_function  |---+----->|assemble_task |---+|validate_task  |
+-----------+   |    +----------------+   |      +--------------+   |+------+--------+
                +--->+----------------+   |                         |       |
                |    |copy_to_workspace|  |                         | +-----v--------+
                |    +----------------+   |                         | |optimize_task  |
                |                         |                         | +-----+---------+
                |    +----------------+   |                         |       |
                +--->|trace_deps      |---+                         | +-----v--------+
                     +----------------+                             | |validate_task  |
                                                                    | +-----+---------+
                                                                    |       |
                                                                    |+------v-----------+
                                                                    ||test_with_reference|
                                                                    |+------------------+
```

**Can be parallelized**: multiple `grep_search`, multiple `copy_to_workspace`, multiple `read_function`, simultaneous `write_file` calls.

**Must be serial**: `grep_search` -> `copy_to_workspace` -> `read_function` -> `trace_dependencies` -> `assemble_task` -> `validate_task` -> `optimize_task` -> `validate_task` -> `test_with_reference` -> `finish`.

## Message Management

TaskConstructor uses full chat completion history. To control context length, `_manage_history()` implements compression:

- **Trigger**: message count exceeds `MAX_MESSAGES=50`
- **Preserved**: system prompt + initial user message + compressed summary + recent messages
- **Strategy**: Middle messages are extracted as `[Operation History Summary]`, preserving key info (action + thought summary, workspace file references)
- **Effect**: In a 37-step task, max prompt was ~69KB, reduced to ~64KB after compression

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
import asyncio
from akg_agents.op.agents.kernel_agent import KernelAgent

async def main():
    # Note: KernelAgent.run() only accepts user_input.
    # source_path should be included in the description;
    # the LLM extracts it when calling the tool.
    agent = KernelAgent(task_id="test_001")
    result = await agent.run(
        user_input="In /path/to/pytorch, find torch._chunk_cat, generate triton kernel",
    )

asyncio.run(main())
```

KernelAgent -> LLM decides -> `call_task_constructor(user_input=..., source_path=...)` -> ToolExecutor -> `TaskConstructor.run()`

## Skill System

### Skill File

Located at `op/resources/skills/task-constructor/SKILL.md`

### Resources

| File | Description | Access Method |
|------|-------------|---------------|
| `references/kernelbench-format.md` | KernelBench format spec | `read_file` tool |
| `references/assembly-strategies.md` | Assembly strategy guide | `read_file` tool |
| `scripts/validate_kernelbench_task.py` | Format validation script | `execute_script` tool |

Resources are not auto-loaded by SkillRegistry. They are documented in SKILL.md and accessed by the LLM on-demand via tools.

## Logging

Each run generates logs under `~/.akg/task_constructor/<timestamp>/logs/`:

| File | Content |
|------|---------|
| `system_prompt.txt` | Full system prompt |
| `initial_message.txt` | Initial user message |
| `session.log` | Text-format session log (LLM response summaries per step) |
| `messages.jsonl` | Structured step records (action/thought/result) |
| `prompt_final.json` | Complete messages list sent to LLM at last step |
| `result.json` | Final result |
| `task_output.py` | Generated task code |

## Directory Structure

```
python/akg_agents/op/
+-- agents/
|   +-- task_constructor.py          # Main agent (ReAct loop, SessionLogger)
+-- tools/task_constructor/
|   +-- __init__.py                  # Entry point, triggers tool registration
|   +-- tool_registry.py             # TaskToolRegistry (register/execute/list)
|   +-- code_tools.py                # Code tool registration entry
|   +-- file_tools.py                # File operation tools
|   +-- path_utils.py                # Shared path resolution
|   +-- ast_utils.py                 # AST parsing
|   +-- code_cleanup.py              # Code cleanup
|   +-- assembly.py                  # Task assembly/validation
|   +-- execution.py                 # Code execution/testing
+-- resources/
    +-- prompts/task_constructor/
    |   +-- system_prompt.j2         # Jinja2 prompt template
    +-- skills/task-constructor/
        +-- SKILL.md                 # Skill description
        +-- references/              # Reference docs (read via read_file)
        +-- scripts/                 # Validation scripts (run via execute_script)
```

## FAQ

**Q: Why do some tasks need 30+ steps?**  
A: Complex operators (e.g., `_chunk_cat`) require multi-level dependency tracing, external function inlining, and signature compatibility fixes. Typical simple tasks complete in 15-20 steps.

**Q: validate_task fails but the code looks correct?**  
A: validate_task runs in an isolated subprocess that may lack certain modules. Use `run_code` for manual testing with more detailed error output.

**Q: How to view the full LLM prompt?**  
A: After a run, check `~/.akg/task_constructor/<timestamp>/logs/prompt_final.json` for the complete messages list sent to the LLM at the last step. `system_prompt.txt` and `initial_message.txt` contain the system prompt and initial message separately.

**Q: How to pass source_path when calling via KernelAgent?**  
A: Include the path in the `user_input` string, e.g. "In /path/to/repo, find...". KernelAgent's LLM will extract the `source_path` parameter when calling the `call_task_constructor` tool.
