[中文版](./README_CN.md)

<div align="center">
  <img src="./akg_agents_logo.jpg" alt="AIKG Logo" width="400">
</div>

<div align="center">

# AKG Agents

</div>

<details>
<summary><b>📋 Table of Contents</b></summary>

- [AKG Agents](#akg-agents)
  - [📘 1. Project Overview](#-1-project-overview)
  - [🗓️ 2. Changelog](#️-2-changelog)
  - [🛠️ 3. Quick Start](#️-3-quick-start)
    - [Installation](#installation)
    - [Configure LLM](#configure-llm)
    - [Launch AKG\_CLI](#launch-akg_cli)
    - [Usage](#usage)
  - [▶️ 4. Tutorial Examples](#️-4-tutorial-examples)
  - [📐 5. Design Documentation](#-5-design-documentation)

</details>

## 📘 1. Project Overview
**AKG Agents** is an LLM-powered multi-agent collaboration framework for AI Infra and high-performance computing, aimed at boosting the development and optimization efficiency of high-performance code through intelligent agent collaboration.

The framework provides a complete agent infrastructure: ReAct Agent base classes, extensible **Skill / Tools / SubAgent** mechanisms, LangGraph workflow orchestration, tree-based Trace system, and a unified configuration and registry. Developers can rapidly build, compose, and deploy intelligent agents tailored to diverse tasks.

The current production scenario is **AI kernel code generation**: leveraging LLM planning and multi-agent collaboration to automate multi-backend, multi-DSL high-performance kernel generation and optimization. Future extensions will cover kernel migration, performance tuning, code refactoring, and more AI Infra related scenarios.

## 🗓️ 2. Changelog
- 2026-02-10: Core framework refactored (v2). Decoupled general-purpose Agent capabilities from kernel-specific logic to build a reusable multi-agent collaboration framework. See [Architecture](./docs/v2/Architecture.md), [Agent System](./docs/v2/AgentSystem.md), [Skill System](./docs/v2/SkillSystem.md), [Workflow](./docs/v2/Workflow.md), [Trace System](./docs/v2/Trace.md), [Configuration](./docs/v2/Configuration.md).
- 2025-12-01: Introduced LangGraph for task orchestration. New `LangGraphTask` replaces original `Task Orchestration` scheme. See [Workflow Documentation](./docs/v2/Workflow.md).
- 2025-11-25: Supported service architecture, including `client-server-worker` separation architecture. See [Service Architecture Documentation](./docs/v1/ServerArchitecture.md).
- 2025-10-14: Supported TileLang_CUDA backend code generation. See [Benchmark Results](./docs/v1/DSLBenchmarkResults202509.md).
- 2025-09-26: Supported CUDA C and CPP backend code generation. See [Benchmark Results](./docs/v1/DSLBenchmarkResults202509.md).
- 2025-09-14: Updated KernelBench Level1 kernel generation success rate. See [Benchmark Results](./docs/v1/BenchmarkResults202509.md).
- 2025-08-12: Supported "Doc-Driven Integration" (now replaced by [Skill System](./docs/v2/SkillSystem.md)).
- 2025-06-27: Initial AIKG release with code generation support for Triton and SWFT backends.


## 🛠️ 3. Quick Start

### Installation
```bash
# 1. Environment setup (optional, recommended Python 3.10/3.11/3.12)
conda create -n akg_agents python=3.11
conda activate akg_agents

# 2. Clone the repository
git clone https://gitcode.com/mindspore/akg.git -b br_agents
cd akg

# 3. Install dependencies
pip install -r akg_agents/requirements.txt

# 4. Install AIKG
pip install -e ./akg_agents --no-build-isolation

# 5. Initialize third-party submodules (KernelBench, etc., as needed)
git submodule update --init "akg_agents/thirdparty/*"
```

### Configure LLM

Copy the example config to `~/.akg/settings.json` and fill in your API Key and model info:

```bash
mkdir -p ~/.akg
cp akg_agents/examples/settings.example.json ~/.akg/settings.json
```

A minimal configuration only requires one model (auto-applies to all levels):

```json
{
  "models": {
    "standard": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "YOUR_API_KEY",
      "model_name": "deepseek-chat",
      "thinking_enabled": true
    }
  },
  "default_model": "standard"
}
```

> For per-level model configuration (`complex` / `standard` / `fast`), Embedding/RAG setup, or environment variable usage, see [Configuration Documentation](./docs/v2/Configuration.md) and the full example [`settings.example.json`](./examples/settings.example.json).

### Backend Dependencies

The `br_agents` branch currently supports the following three DSLs. Other backends are pending adaptation:

| Platform | Backend (DSL) | Reference Link |
|----------|---------------|----------------|
| Huawei Atlas A2 Training Series | Triton | https://gitee.com/ascend/triton-ascend |
| NVIDIA GPU | Triton | https://github.com/triton-lang/triton |
| CPU (x86_64) | C++ | GCC / Clang |

### Launch & Usage

Taking the kernel code generation task (`akg_cli op`) as an example:

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 \
  --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100
# akg_cli op --framework torch --backend cuda --arch a100 \
#   --dsl triton_cuda --devices 0,1,2,3,4,5,6,7
```

Once launched, you can interact in the following ways:

1. **Describe your requirement**: For example, "help me generate a relu kernel"
2. **Provide code**: Paste KernelBench-style PyTorch code, and AIKG will automatically generate the corresponding DSL kernel implementation and verify correctness

> `akg_cli` also supports other task types. For full usage details, see [AKG CLI Documentation](./docs/v2/AKG_CLI.md).


## ▶️ 4. Tutorial Examples

<details open>
<summary><b>examples/ directory</b></summary>

| Example | Category | Description |
|---------|----------|-------------|
| `kernel_related/run_torch_npu_triton_single.py` | Kernel | Single kernel generation (Torch + Triton Ascend) |
| `kernel_related/run_torch_adaptive_search_triton_ascend.py` | Kernel | UCB adaptive search (Torch + Triton Ascend) |
| `kernel_related/run_torch_evolve_triton_ascend.py` | Kernel | Evolutionary kernel optimization (Torch + Triton Ascend) |
| `kernel_related/run_triton_to_torch_single.py` | Kernel | Single kernel generation (Torch + Triton CUDA) |
| `kernel_related/run_cudac_to_torch_single.py` | Kernel | Single kernel generation (Torch + CUDA C) |
| `kernel_related/run_torch_cpu_cpp_single.py` | Kernel | Single kernel generation (Torch + CPP) |
| `kernel_related/run_torch_evolve_triton.py` | Kernel | Evolutionary kernel optimization (Torch + Triton CUDA) |
| `kernel_related/run_cuda_to_ascend_conversion.py` | Kernel | CUDA to Ascend kernel conversion |
| `kernel_related/run_cuda_to_ascend_evolve.py` | Kernel | CUDA to Ascend evolutionary optimization |
| `kernel_related/run_kernel_agent.py` | Kernel | KernelAgent (ReAct Agent) interactive invocation |
| `kernel_related/run_client_server_worker.py` | Kernel | Client-Server distributed execution |
| `kernel_related/kernel_profile.py` | Kernel | Kernel performance profiling |
| `run_skill/` | Skill | Skill loading, registry, hierarchy, versioning, installation, LLM selection examples |
| `build_a_simple_react_agent/` | Framework | Build a custom ReAct Agent using the framework |
| `build_a_simple_workflow/` | Framework | Build a custom LangGraph-based Workflow |
| `settings.example.json` | Config | Full `settings.json` configuration template |

</details>


## 📐 5. Design Documentation

> Start with [Architecture](./docs/v2/Architecture.md) for an overview, then read [Workflow](./docs/v2/Workflow.md) and [Skill System](./docs/v2/SkillSystem.md) to understand the core mechanisms.

### Core Framework
- **[Architecture](./docs/v2/Architecture.md)** - Overall architecture and module overview
- **[Agent System](./docs/v2/AgentSystem.md)** - Agent base classes, ReAct Agent, registry
- **[Skill System](./docs/v2/SkillSystem.md)** - Skill management and dynamic knowledge injection
- **[Tools](./docs/v2/Tools.md)** - Tool execution framework, built-in tools, domain tools
- **[Workflow](./docs/v2/Workflow.md)** - LangGraph-based workflow orchestration
- **[Trace System](./docs/v2/Trace.md)** - Tree-based inference tracing (multi-fork, checkpoint resume)
- **[Configuration](./docs/v2/Configuration.md)** - Unified configuration management (settings.json / env vars)
- **[LLM](./docs/v2/LLM.md)** - LLM provider, client, embedding

### Scenarios
- **[Kernel Agent](./docs/v2/KernelAgent.md)** - Multi-backend, multi-DSL kernel code generation and optimization (`akg_cli op`)

### CLI
- **[AKG CLI](./docs/v2/AKG_CLI.md)** - Command-line tool usage guide

### Contributing
- **[Skill Contribution Guide](./docs/v2/SkillContributionGuide.md)** - How to contribute new Skills

### Additional Modules (v1 Documentation)
- **[Database](./docs/v1/Database.md)** - Database module
- **[RAG](./docs/v1/RAG.md)** - Vector retrieval-augmented generation
- **[RAG Usage Guide](./docs/v1/RAG_Usage.md)** - RAG configuration and usage tutorial
- **[Server Architecture](./docs/v1/ServerArchitecture.md)** - Service architecture (Client-Server-Worker)
- **[TaskPool](./docs/v1/TaskPool.md)** - Task pool management
- **[DevicePool](./docs/v1/DevicePool.md)** - Device pool management
