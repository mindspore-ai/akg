[中文版](./CN/AKG_CLI.md)

# AKG CLI

## 1. Overview

`akg_cli` is the command-line interface for AKG Agents, providing interactive access to all framework capabilities.

> For installation and basic configuration, see [README](../README.md#️-3-quick-start).

## 2. Commands

| Command | Description |
|---------|-------------|
| `akg_cli op` | Kernel Agent — multi-backend, multi-DSL kernel generation. See [Kernel Agent](./KernelAgent.md). |
| `akg_cli common` | Common Agent — general-purpose ReAct agent (demo). |
| `akg_cli worker --start` | Start Worker Service for distributed execution. |
| `akg_cli worker --stop` | Stop Worker Service. |
| `akg_cli sessions` | List all sessions with trace history. |
| `akg_cli resume <session_id>` | Resume a previous session. |
| `akg_cli list` | List supported subcommands. |

## 3. akg_cli op

The primary command for AI kernel code generation.

### Required Parameters

| Parameter | Description | Examples |
|-----------|-------------|----------|
| `--framework` | Compute framework | `torch`, `mindspore` |
| `--backend` | Hardware backend | `ascend`, `cuda`, `cpu` |
| `--arch` | Hardware architecture | `ascend910b2`, `ascend910b4`, `a100`, `x86_64` |
| `--dsl` | Target DSL | `triton_ascend`, `triton_cuda`, `cuda`, `tilelang_cuda`, `cpp` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--intent` | `None` | Provide requirement text directly (skip interactive prompt) |
| `--task-file` | `None` | Read task description file (KernelBench format) |
| `--devices` | `None` | Local device list, comma-separated (e.g., `0,1,2,3`) |
| `--worker-url` | `None` | Worker Service URLs, comma-separated |
| `--stream/--no-stream` | `--stream` | Enable/disable LLM streaming output |
| `--rag/--no-rag` | `--no-rag` | Enable/disable RAG retrieval |
| `--resume` | `None` | Resume a previous session by ID |
| `-y` | `False` | Auto-confirm all prompts |

### Examples

```bash
# Ascend 910B2 with Triton
akg_cli op --framework torch --backend ascend --arch ascend910b2 \
  --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100 with Triton
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0,1,2,3,4,5,6,7

# With direct intent
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0 --intent "Generate a relu kernel"

# With KernelBench task file
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0 --task-file path/to/task.py
```

## 4. Worker Service

For distributed execution across multiple machines:

```bash
# Start worker on each machine
akg_cli worker --start --port 9001

# Connect from client
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --worker-url machine1:9001,machine2:9002
```

## 5. Session Management

```bash
# List all sessions
akg_cli sessions list

# Resume a session
akg_cli resume <session_id>
```
