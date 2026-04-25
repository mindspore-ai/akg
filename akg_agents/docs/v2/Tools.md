[中文版](./CN/Tools.md)

# Tools

## 1. Overview

The Tools module provides the tool execution framework for AKG Agents. Agents use tools to interact with the external environment — reading files, running scripts, verifying kernels, etc.

Key components:

- **ToolExecutor** — Executes tools by name, resolves arguments, captures errors, persists results
- **Basic Tools** — Built-in general-purpose tools (file I/O, shell, code checking)
- **Domain Tools** — Scenario-specific tools (kernel verification, performance profiling)
- **Tool Configuration** — YAML-based tool definitions (`tools.yaml`, `domain_tools.yaml`)

## 2. ToolExecutor

`ToolExecutor` is the central tool execution engine used by `ReActAgent`. It:

1. Receives tool name and arguments from LLM output
2. Resolves argument expressions (e.g., `read_json_file('path')['key']`)
3. Dispatches to the appropriate tool function
4. Captures errors (WARNING+ logs, stderr) during execution
5. Persists results to the trace node path

### Key Features

- **Argument Resolution**: Supports expressions like `read_json_file('result.json')['output']` in tool arguments via `arg_resolver`
- **Error Capture**: Captures WARNING+ level logs and stderr during tool execution, injects concise error summaries into results
- **Agent/Workflow Dispatch**: Can dispatch to registered agents (`call_agent` type) or workflows (`call_workflow` type) in addition to regular tools

## 3. Built-in Tools (basic_tools)

| Tool | Description | Required Parameters |
|------|-------------|---------------------|
| `read_file` | Read file contents. Supports relative and absolute paths. | `file_path` |
| `write_file` | Write content to a file. Supports auto-naming by op_name. | `content` |
| `ask_user` | Ask the user a question and wait for reply. | `message` |
| `check_python_code` | Check Python syntax and auto-format with autopep8. | `file_path` |
| `check_markdown` | Check Markdown format with markdownlint-cli. | `file_path` |
| `execute_script` | Execute a shell or Python script. | `script_path` |

### write_file Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | File content to write |
| `file_path` | string | No | Explicit file path. If not provided, auto-generated from `op_name` and `file_type` |
| `op_name` | string | No | Kernel name for auto-naming |
| `file_type` | string | No | File type/extension for auto-naming |
| `encoding` | string | No | File encoding (default: `"utf-8"`) |
| `overwrite` | bool | No | Whether to overwrite existing file (default: `True`) |

### execute_script Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `script_path` | string | Yes | Path to the script to execute |
| `args` | list | No | Command-line arguments to pass to the script |
| `stdin_input` | string | No | Standard input to feed to the script |
| `timeout` | int | No | Execution timeout in seconds (default: `300`) |
| `working_dir` | string | No | Working directory for script execution |

### Return Format

All basic tools return a standardized dict:

```python
{
    "status": "success" | "error",
    "output": "...",           # Tool output
    "error_information": "..."  # Error details (empty on success)
}
```

## 4. Domain Tools (domain_tools)

Domain tools are scenario-specific tools for kernel generation and optimization.

### verify_kernel

Verify the correctness of generated kernel code by comparing outputs between the framework implementation (e.g., PyTorch) and the generated implementation (e.g., Triton).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_code` | string | Yes | Framework implementation code (must contain `class Model`) |
| `generated_code` | string | Yes | Generated code to verify (must contain `class ModelNew`) |
| `op_name` | string | Yes | Kernel name (e.g., `"relu"`, `"matmul"`) |
| `task_id` | string | No | Task ID (default: `"default_task"`) |
| `device_id` | int | No | Device ID (default: `0`) |
| `timeout` | int | No | Timeout in seconds (default: `300`) |
| `cur_path` | string | No | Current trace node path (default: `""`) |
| `framework` | string | No | Compute framework (default: `"torch"`) |
| `backend` | string | No | Hardware backend (default: `"cuda"`) |
| `arch` | string | No | Hardware architecture (default: `"a100"`) |
| `dsl` | string | No | Target DSL (default: `"triton"`) |

### profile_kernel

Performance profiling for kernel code. Returns execution times and speedup ratio.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_code` | string | Yes | Framework implementation code |
| `generated_code` | string | Yes | Generated code to profile |
| `op_name` | string | Yes | Kernel name |
| `task_id` | string | No | Task ID (default: `"default_task"`) |
| `device_id` | int | No | Device ID (default: `0`) |
| `run_times` | int | No | Number of profiling runs (default: `50`) |
| `warmup_times` | int | No | Number of warmup runs (default: `5`) |
| `cur_path` | string | No | Current trace node path (default: `""`) |
| `framework` | string | No | Compute framework (default: `"torch"`) |
| `backend` | string | No | Hardware backend (default: `"cuda"`) |
| `arch` | string | No | Hardware architecture (default: `"a100"`) |
| `dsl` | string | No | Target DSL (default: `"triton"`) |

> Note: `profile_kernel` only measures performance — it does not verify correctness.

## 5. Tool Configuration

Tools are defined in YAML configuration files:

- `tools.yaml` — Basic tool definitions
- `domain_tools.yaml` — Domain-specific tool definitions

### Configuration Format

```yaml
tools:
  read_file:
    type: "basic_tool"
    function:
      name: "read_file"
      description: "Read file contents."
      parameters:
        type: "object"
        properties:
          file_path:
            type: "string"
            description: "File path"
        required: ["file_path"]

  verify_kernel:
    type: "domain_tool"
    function:
      name: "verify_kernel"
      description: "Verify generated kernel code correctness."
      parameters:
        type: "object"
        properties:
          task_code:
            type: "string"
            description: "Framework implementation code"
          generated_code:
            type: "string"
            description: "Generated code to verify"
          op_name:
            type: "string"
            description: "Kernel name"
        required: ["task_code", "generated_code", "op_name"]
```

### Tool Types

| Type | Description |
|------|-------------|
| `basic_tool` | General-purpose tool, dispatched to `basic_tools` module |
| `domain_tool` | Domain-specific tool, dispatched to `domain_tools` module |
| `call_agent` | Dispatches to a registered agent (via `AgentRegistry`) |
| `call_workflow` | Dispatches to a registered workflow (via `WorkflowRegistry`) |

## 6. Argument Resolver

The `arg_resolver` module enables dynamic argument resolution in tool calls. When the LLM generates tool arguments containing expressions, they are resolved before execution.

### Supported Expressions

```python
# Read a JSON file and extract a field
read_json_file('path/to/result.json')['output']

# Nested access
read_json_file('path/to/result.json')['data']['code']
```

This allows agents to chain tool results — for example, reading the output of a previous tool's `result.json` and passing it as input to the next tool.
