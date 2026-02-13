[中文版](./CN/KernelAgent.md)

# AKG Kernel Agent

## 1. Overview

The AKG Kernel Agent is the first production scenario of AKG Agents, focused on **multi-backend, multi-DSL high-performance kernel code generation and optimization**.

- **CLI entry**: `akg_cli op`
- **Domain**: Kernel code generation for AI accelerators

## 2. Supported Backends and DSLs

| Platform | Backend | DSL | Example |
|----------|---------|-----|---------|
| Huawei Atlas A2 Training Series | Ascend | Triton Ascend | `--backend ascend --dsl triton_ascend` |
| NVIDIA GPU | CUDA | Triton CUDA | `--backend cuda --dsl triton_cuda` |
| NVIDIA GPU | CUDA | CUDA C | `--backend cuda --dsl cuda` |
| NVIDIA GPU | CUDA | TileLang CUDA | `--backend cuda --dsl tilelang_cuda` |
| CPU | CPU | C++ | `--backend cpu --dsl cpp` |

## 3. Built-in Workflows

The Kernel Agent supports multiple workflow strategies:

| Workflow | Description |
|----------|-------------|
| **Default** | Full pipeline: Designer → Coder ↔ Verifier |
| **CoderOnly** | Code generation only (skip design phase) |
| **Evolve** | Evolutionary algorithm-based kernel optimization |
| **AdaptiveSearch** | UCB-based asynchronous pipeline search |
| **KernelGenOnly** | Kernel generation without verification |
| **VerifierOnly** | Verification only (for pre-existing code) |

## 4. Core Agents

### KernelDesigner

Algorithm sketch design agent. Analyzes the kernel requirement and produces a high-level algorithm design with optimization hints.

- Skill-based: dynamically injects relevant domain knowledge
- Supports DSL-specific design patterns

### KernelGen

Kernel code generation agent. Takes the algorithm design and generates executable kernel code in the target DSL.

- Skill-based: uses DSL-specific coding skills
- Callable as a tool by other agents

### TaskConstructor

Standardized task builder agent. Extracts and standardizes kernel definitions from user input (e.g., PyTorch code) into a structured task format.

## 5. Workflow: Default Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Designer   │────▶│    Coder     │────▶│   Verifier   │
│              │     │  (KernelGen) │     │              │
│ Algorithm    │     │ Code Gen     │     │ Correctness  │
│ Design       │     │              │◀────│ Check        │
└──────────────┘     └──────────────┘     └──────────────┘
                           │                     │
                           │                     ▼
                           │              ┌──────────────┐
                           │              │   Profiler   │
                           │              │ Performance  │
                           │              │ Analysis     │
                           └──────────────┴──────────────┘
```

1. **Designer** analyzes the kernel requirement and produces an algorithm sketch
2. **Coder** generates kernel code based on the design
3. **Verifier** checks correctness by comparing against the framework implementation
4. If verification fails, the Coder receives error feedback and retries
5. **Profiler** measures performance (execution time, speedup ratio)

## 6. DSL Configuration

Each DSL backend has a YAML configuration file controlling workflow behavior:

```yaml
# Key configuration fields
agent_model_config:
  kernel_designer: "complex"
  kernel_gen: "standard"

log_dir: "logs/"
default_workflow: "default"

profile_settings:
  run_times: 50
  warmup_times: 5

verify_timeout: 300
```

## 7. Quick Start

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 \
  --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0,1,2,3,4,5,6,7
```

After launch, you can:
1. Describe what you need: "Generate a relu kernel"
2. Paste KernelBench-style PyTorch code for conversion

For more CLI details, see [AKG CLI](./AKG_CLI.md).
