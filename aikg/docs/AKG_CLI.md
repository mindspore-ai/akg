# AKG CLI

AKG CLI (`akg_cli`) is the command-line interface for AI Kernel Generator. This document focuses on the two most common flows:

- Start a Worker Service: `akg_cli worker --start`
- Generate an operator: `akg_cli op ...`

## 0. Installation

First, install `akg_cli` by running the following command in the `aikg` directory:

```bash
pip install -e .
```

## 1. Start a Worker Service

A Worker Service provides the backend runtime for kernel generation. Start it before running `akg_cli op`.

```bash
akg_cli worker --start \
  --backend cuda \
  --arch a100 \
  --devices 0 \
  --host 127.0.0.1 \
  --port 9001
```

Notes:

- `--backend` supports `cuda` or `ascend`. If omitted, defaults to `WORKER_BACKEND` or `cuda`.
- `--arch` defaults to `WORKER_ARCH` or `a100`.
- `--devices` is a comma-separated list of local device IDs (e.g., `0` or `0,1,2`). It must be non-empty, contain no duplicates, and have no negative numbers.
- `--host` defaults to `0.0.0.0`, `--port` defaults to `9001`.

Stop the service when needed:

```bash
akg_cli worker --stop --port 9001
```

## 2. Generate an Operator

Use `akg_cli op` to start operator generation. The target config must be provided explicitly from CLI.

```bash
akg_cli op \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --worker-url 127.0.0.1:9001
```

Optional parameters:

- `--intent "..."` to provide requirements directly (skip interactive prompt).
- `--worker-url` accepts multiple worker addresses separated by commas.
- `--devices` registers local devices to the local server (no worker URL needed). It is mutually exclusive with `--worker-url` and only works with the local server.
- `--stream/--no-stream` to enable/disable streaming output.
- `--yes` to auto-confirm all prompts.

Examples:

```bash
akg_cli op \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --worker-url 127.0.0.1:9001 \
  --intent "implement fused softmax kernel for input [batch, head, seq, dim]"
```

```bash
akg_cli op \
  --framework mindspore \
  --backend ascend \
  --arch ascend910b4 \
  --dsl triton_ascend \
  --worker-url 127.0.0.1:9001,127.0.0.1:9002
```

```bash
akg_cli op \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --devices 0
```

## 3. TUI Basics

The TUI is the default interactive interface for `akg_cli op`. It is split into several panels:

- Top: task tabs.
  - The `main` tab is the primary dialogue.
  - Each sub-agent runs in its own tab; switch tabs to view their outputs.
- Left: chat log (main output stream).
- Right: task info panel, workflow panel, and trace list.
- Bottom: input box.

Focus & navigation:

- `Tab` cycles focus between panels.
- In the trace list, select an item to jump the chat log to the corresponding position.
- `F8` switches to the next concurrent task (when task tabs are available).

Common shortcuts:

- `Ctrl+C`: quit.
- `Ctrl+E`: scroll to bottom.
- `Enter`: submit input (in the input box).
- `Ctrl+J`: insert newline (in the input box).
