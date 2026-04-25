# AKG CLI

AKG CLI (`akg_cli`) is the command-line interface for AI Kernel Generator. This document focuses on the two most common flows:

- Start a Worker Service: `akg_cli worker --start`
- Generate an operator: `akg_cli op ...`

How to get help:

```bash
akg_cli --help
akg_cli op --help
akg_cli worker --help
```

## 0. Installation

First, install `akg_cli` by running the following command in the `aikg` directory:

```bash
pip install -e .
```

## 1. Start a Worker Service

A Worker Service provides the backend runtime for kernel generation. Start it before running `akg_cli op`.

```bash
# Ascend 910B2
akg_cli worker --start --backend ascend --arch ascend910b2 --devices 0 --host 127.0.0.1 --port 9001

# CUDA A100: --backend cuda --arch a100
# akg_cli worker --start --backend cuda --arch a100 --devices 0 --host 127.0.0.1 --port 9001
```

Notes:

- `--backend` supports `cuda` or `ascend`. If omitted, defaults to `WORKER_BACKEND` or `cuda`.
- `--arch` defaults to `WORKER_ARCH` or `a100`.
- `--devices` is a comma-separated list of local device IDs (e.g., `0` or `0,1,2`). It must be non-empty, contain no duplicates, and have no negative numbers.
- `--host` defaults to `0.0.0.0`, `--port` defaults to `9001`. For IPv6-only machines, use `--host ::` (dual-stack/IPv6 bind).

Stop the service when needed:

```bash
akg_cli worker --stop --port 9001
```

## 2. Generate an Operator

Use `akg_cli op` to start operator generation. The target config must be provided explicitly from CLI.

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001

# CUDA A100: --backend cuda --arch a100 --dsl triton_cuda
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001
```

Optional parameters:

- `--intent "..."` to provide requirements directly (skip interactive prompt).
- `--worker-url` accepts multiple worker addresses separated by commas. The CLI also accepts both `--worker-url` and `--worker_url`.
- Choose exactly one of the following:
  - Use `--worker-url/--worker_url` when you want to run with remote Worker Service(s).
  - Use `--devices` when you want to run on local devices.
  - `--devices` and `--worker-url/--worker_url` are mutually exclusive.
- `--output-path` sets the base directory for `saved_verifications` (default: current working directory where `akg_cli` is started).
- `--stream/--no-stream` to enable/disable streaming output (default: `--stream`).
- `--rag/--no-rag` to enable/disable RAG (default: `--no-rag`).
- `--yes/-y` to auto-confirm all prompts.

Tip: for a mostly non-interactive run, use `--intent "..." --yes` (best-effort; the CLI may still ask for input in some cases).

Examples:

```bash
# Ascend 910B2 with intent
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001 --intent "implement fused softmax kernel for input [batch, head, seq, dim]"

# CUDA A100 with intent
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001 --intent "implement fused softmax kernel for input [batch, head, seq, dim]"
```

Multiple workers example:

```bash
# Ascend 910B2 with multiple workers
akg_cli op --framework mindspore --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001,127.0.0.1:9002
```

Local devices example (no worker URL needed):

```bash
# Ascend 910B2 with local devices
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --devices 0

# CUDA A100 with local devices
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --devices 0
```

IPv6-only example (note the brackets):

```bash
akg_cli worker --start --host :: --port 9001
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url [::1]:9001
```

## 3. Interactive CLI Basics

`akg_cli op` runs an interactive prompt loop. You can iterate with multiple rounds of input and generation in the same session.

Input:

- `Enter`: submit the current input.
- `Ctrl+J`: insert a newline (multi-line input is supported).

Exit / cancel:

- When idle, press `Ctrl+C` to exit.
- While generating, press `Ctrl+C` to cancel the current round. You can then continue with the next round, or press `Ctrl+C` again to exit.

Panel:

- `F2`: show/hide the information panel.
- The panel is used to show the current task status, output/save path, and recent top implementation summaries.

History:

- Use the Up/Down arrows to browse input history.
- History is saved to `~/.akg_cli_history`.
