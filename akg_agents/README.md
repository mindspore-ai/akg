[中文版](./README_CN.md)

<div align="center">
  <img src="./akg_agents_logo.jpg" alt="AIKG Logo" width="400">
</div>

<div align="center">

# AKG Agents (AKG Agents)

</div>

<details>
<summary><b>📋 Table of Contents</b></summary>

- [AKG Agents (AKG Agents)](#akg-agents-akg-agents)
  - [📘 1. Project Overview](#-1-project-overview)
  - [🗓️ 2. Changelog](#️-2-changelog)
  - [🛠️ 3. AKG\_CLI Quick Start](#️-3-akg_cli-quick-start)
    - [Basic Installation](#basic-installation)
    - [Configure LLM](#configure-llm)
    - [Launch AKG\_CLI](#launch-akg_cli)
    - [Usage](#usage)
  - [⚙️ 4. Configuration](#️-4-configuration)
    - [Configuration Quick Guide](#configuration-quick-guide)
      - [Step 1: Basic Environment Configuration](#step-1-basic-environment-configuration)
        - [API and Model Configuration](#api-and-model-configuration)
        - [Third-party Dependencies](#third-party-dependencies)
      - [Step 2: Backend Dependencies Configuration](#step-2-backend-dependencies-configuration)
      - [Step 3: Optional Tools Configuration](#step-3-optional-tools-configuration)
        - [Embedding Model Configuration (RAG)](#embedding-model-configuration-rag)
  - [▶️ 5. Tutorial Examples](#️-5-tutorial-examples)
  - [📐 6. Design Documentation](#-6-design-documentation)
    - [Core Framework](#core-framework)
    - [Core Components (Agents & Search Optimization)](#core-components-agents--search-optimization)
    - [Knowledge & Data](#knowledge--data)
    - [Service Architecture](#service-architecture)
    - [Backend Support](#backend-support)

</details>

## 📘 1. Project Overview
AKG Agents is a multi-agent collaboration framework powered by Large Language Models (LLMs).
The framework provides a general-purpose agent orchestration engine (`core_v2`): including ReAct Agent base classes, Skill-based dynamic knowledge injection, LangGraph workflows, tree-based Trace system, and other foundational capabilities for building various AI-assisted scenarios.
The first production scenario is **AI Kernel Generation** (`op`): using LLM planning and multi-agent collaboration to automate multi-backend, multi-DSL AI kernel generation and optimization. Future extensions will cover documentation, refactoring, testing, and other general-purpose scenarios.

<div align="center" style="background-color:white">
  <img src="./akg_agents.png" alt="AIKG Architecture" width="600">
</div>

## 🗓️ 2. Changelog
- 2026-02-10: Core framework Refactored. Decoupled general-purpose Agent capabilities from operator-specific logic to build a reusable multi-agent collaboration framework. Key changes include: unified configuration management ([`settings.json`](./examples/settings.example.json)), Agent base classes & registry (`AgentBase` / `ReActAgent` / `AgentRegistry`), [Skill-based dynamic knowledge injection](./docs/SkillSystem.md), [tree-based Trace system](./docs/Trace.md), [LangGraph workflows](./docs/Workflow.md), OpenAI-compatible [Embedding/RAG](./docs/RAG.md), and [Database base class decoupling](./docs/Database.md). See [Refactor Migration Guide](./docs/Refactor.md) for details.
- 2025-12-01: Introduced LangGraph for task orchestration. New `LangGraphTask` replaces original `Task Orchestration` scheme. See [Workflow Documentation](./docs/Workflow.md).
- 2025-11-25: Supported service architecture, including `client-server-worker` separation architecture. See [Service Architecture Documentation](./docs/ServerArchitecture.md).
- 2025-10-14: Supported TileLang_CUDA backend code generation. See [Benchmark Results](./docs/DSLBenchmarkResults202509.md).
- 2025-09-26: Supported CUDA C and CPP backend code generation. See [Benchmark Results](./docs/DSLBenchmarkResults202509.md).
- 2025-09-14: Updated KernelBench Level1 kernel generation success rate. See [Benchmark Results](./docs/BenchmarkResults202509.md).
- 2025-08-12: Supported "Doc-Driven Integration" (now replaced by [Skill System](./docs/SkillSystem.md)).
- 2025-06-27: Initial AIKG release with code generation support for Triton and SWFT backends.


## 🛠️ 3. AKG_CLI Quick Start

### Basic Installation
```bash
# 1. Environment setup (optional, recommended Python 3.10/3.11/3.12)
# Use conda environment
conda create -n akg_agents python=3.11
conda activate akg_agents

# 2. Clone the repository
git clone https://gitcode.com/mindspore/akg.git -b br_akg_agents
cd akg

# 3. Install dependencies
pip install -r akg_agents/requirements.txt
# pip install -r akg_agents/rag_requirements.txt  # For RAG features (optional)

# 4. Install AIKG
pip install -e ./akg_agents --no-build-isolation

# 5. Setup environment variables
cd ./akg_agents
source env.sh
```

### Configure LLM

We recommend using `settings.json` for configuration (see full example at [`examples/settings.example.json`](./examples/settings.example.json)):

```bash
# Copy the example config to the project config directory
mkdir -p .akg
cp examples/settings.example.json .akg/settings.json
# Edit .akg/settings.json and fill in your API Key and model info
```

`settings.json` supports per-level model configuration (`complex` / `standard` / `fast`):

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
  "embedding": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "YOUR_API_KEY",
    "model_name": "BAAI/bge-large-zh-v1.5"
  },
  "default_model": "standard"
}
```

Environment variables are also supported for quick setup (takes priority over `settings.json`):

```bash
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"
export AKG_AGENTS_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"  # or "disabled"
```

> 💡 Config loading priority (high to low): Environment Variables → `.akg/settings.local.json` → `.akg/settings.json` → `~/.akg/settings.json` → Defaults

### Launch AKG_CLI
```bash
# Ascend 910B2 (--framework torch/mindspore/numpy, --dsl triton_ascend/triton_cuda/cuda/tilelang_cuda, etc.)
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100: --backend cuda --arch a100 --dsl triton_cuda
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --devices 0,1,2,3,4,5,6,7
```

### Usage
After launching AKG_CLI, you can use it in the following ways:

1. **Direct prompts**: For example, "help me generate a relu operator"
2. **Provide code**: Paste existing KernelBench-style code
   - AIKG will first write a baseline torch code for result comparison
   - After verification, you can ask it to generate the target code (the DSL type is determined by the `--dsl` parameter at launch)

> 💡 **Tip**: For more usage examples and detailed instructions, please refer to the [AKG_CLI Documentation](./docs/AKG_CLI.md)



## ⚙️ 4. Configuration

### Configuration Quick Guide

#### Step 1: Basic Environment Configuration

##### API and Model Configuration

AIKG uses a multi-level configuration system for LLM services. We recommend `settings.json`, with environment variable support for backwards compatibility.

**Method 1: `settings.json` (Recommended)**

Copy [`examples/settings.example.json`](./examples/settings.example.json) to `.akg/settings.json` and edit:

```json
{
  "models": {
    "complex": {
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key": "your-api-key",
      "model_name": "Pro/zai-org/GLM-4.7"
    },
    "standard": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "your-deepseek-api-key",
      "model_name": "deepseek-chat",
      "thinking_enabled": true
    },
    "fast": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "your-deepseek-api-key",
      "model_name": "deepseek-chat"
    },
  },
  "embedding": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "your-siliconflow-api-key",
    "model_name": "BAAI/bge-large-zh-v1.5"
  },
  "default_model": "standard"
}
```

**Method 2: Environment Variables (Quick Start / CI)**

```bash
# Single model config (auto-applies to all levels: complex/standard/fast)
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"

# Per-level config (optional)
export AKG_AGENTS_COMPLEX_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_COMPLEX_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_COMPLEX_MODEL_NAME="gpt-4"
```

> 💡 Config loading priority: Environment Variables > `.akg/settings.local.json` > `.akg/settings.json` > `~/.akg/settings.json` > Defaults

More configuration options:
- **Workflow Configuration**: LangGraph-based Python-defined workflows with graph visualization and type-safe state management. See [Workflow Documentation](./docs/Workflow.md).
- **Skill System**: Intelligent skill selection for dynamic domain knowledge injection (replaces old `docs_dir` Doc-Driven approach). See [Skill System Documentation](./docs/SkillSystem.md).

For detailed configuration instructions, please refer to [API Configuration Documentation](./docs/API.md).

##### Third-party Dependencies
This project uses git submodules to manage certain third-party dependencies (e.g., Kernelbench, MultiKernelbench, etc.).

After initial cloning or pulling updates, please use the following command to initialize and download `akg_agents`-related dependencies:
```bash
# Initialize and pull akg_agents-related submodules
git submodule update --init "akg_agents/thirdparty/*"
```

#### Step 2: Backend Dependencies Configuration
Choose the appropriate backend based on your hardware platform:

| Platform | Backend | Reference Link |
|----------|---------|----------------|
| Huawei Atlas A2 Training Series | Triton | https://gitee.com/ascend/triton-ascend |
| NVIDIA GPU | Triton | https://github.com/triton-lang/triton |
| NVIDIA GPU | TileLang | https://github.com/tile-ai/tilelang |
| Huawei Atlas A2 Training Series | TileLang | https://github.com/tile-ai/tilelang |
| NVIDIA GPU | CUDA C/C++ | https://docs.nvidia.com/cuda/ |

#### Step 3: Optional Tools Configuration

##### Embedding Model Configuration (RAG)

The RAG retrieval feature requires an Embedding model for vector representation. The system supports **Remote API** (recommended) and **Local Model** modes.

**Method 1: Remote Embedding API (Recommended)**

Configure the `embedding` field in `settings.json`, or set environment variables:

```bash
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export AKG_AGENTS_EMBEDDING_API_KEY="YOUR_API_KEY"
```

**Method 2: Local HuggingFace Model**

Set the local model path:

```bash
export EMBEDDING_MODEL_PATH="/path/to/your/embedding-model"
```

> 💡 Loading priority: Remote API config → Local `EMBEDDING_MODEL_PATH` → Disable vector retrieval

See [RAG Usage Guide](./docs/RAG_Usage.md) and [RAG Documentation](./docs/RAG.md) for details.

> 💡 **Configuration Tips**: 
> - For detailed API configuration, please refer to [API Documentation](./docs/API.md) 
> - For database configuration, please refer to [Database Documentation](./docs/Database.md)
> - For more configuration options, please refer to the dedicated documentation for each component

## ▶️ 5. Tutorial Examples

Below are common examples in the `examples/` directory:

| Example | Description |
|--------|-------------|
| `run_torch_npu_triton_single.py` | Single operator example (Torch + Triton, Ascend). |
| `run_torch_cpu_cpp_single.py` | CPU C++ single operator example (Torch + CPP). |
| `run_cudac_to_torch_single.py` | CUDA C single operator example (Torch + CUDA C). |
| `run_triton_to_torch_single.py` | Triton CUDA single operator example (Torch + Triton CUDA). |
| `run_kernel_agent.py` | KernelAgent (ReAct Agent) interactive invocation example. |
| `run_torch_evolve_triton.py` | Evolutionary algorithm operator optimization example (Torch + Triton). |
| `run_cuda_to_ascend_conversion.py` | CUDA to Ascend operator conversion example. |
| `run_client_server_worker.py` | Client-Server distributed run example. |
| `kernel_profile.py` | Operator performance Profiling example. |
| `handwrite_optimization_analyzer.py` | Handwrite optimization analyzer example. |

For more getting started steps and parameter notes, please refer to the [Tutorial](./docs/Tutorial.md).


## 📐 6. Design Documentation

> Start with [Workflow Documentation](./docs/Workflow.md) to understand the LangGraph workflow and task orchestration, then read [Skill System](./docs/SkillSystem.md) to learn about knowledge management and dynamic injection.

### Core Framework
- **[Workflow & Task System](./docs/Workflow.md)** - LangGraph-based workflows and task management (`LangGraphTask`)
- **[Trace System](./docs/Trace.md)** - Tree-based inference tracing (supports multi-fork, checkpoint resume)
- **[Skill System](./docs/SkillSystem.md)** - Skill management and dynamic knowledge injection
- **[TaskPool](./docs/TaskPool.md)** - Task pool management
- **[DevicePool](./docs/DevicePool.md)** - Device pool management

### Core Components (Agents & Search Optimization)
- **[KernelDesigner](./docs/KernelDesigner.md)** - Algorithm sketch design Agent (`call_kernel_designer`)
- **[KernelGen](./docs/KernelGen.md)** - Kernel code generation Agent (`call_kernel_gen`)
- **[Verifier](./docs/Workflow.md)** - Code verification node (correctness checking + performance profiling)
- **[Evolve](./docs/Evolve.md)** - Genetic algorithm-based kernel evolution and optimization
- **[Adaptive Search](./docs/Search.md)** - UCB-based asynchronous pipeline search framework
- **[Refactor Migration Guide](./docs/Refactor.md)** - Module migration and refactoring documentation

### Knowledge & Data
- **[Database](./docs/Database.md)** - Database module (base class + operator-specific `CoderDatabase`)
- **[RAG](./docs/RAG.md)** - Vector retrieval-augmented generation module
- **[RAG Usage Guide](./docs/RAG_Usage.md)** - RAG configuration and usage tutorial
- **[Skill Contribution Guide](./docs/SkillContributionGuide.md)** - How to contribute new Skills

### Service Architecture
- **[Server Architecture](./docs/ServerArchitecture.md)** - Service architecture documentation, including Client-Server-Worker architecture, WorkerManager load balancing, convenience function usage, etc.

### Backend Support
- **[Triton Backend (Ascend/CUDA)](./docs/Triton.md)** - Triton compute backend
- **[TileLang Backend (CUDA)](./docs/DSLBenchmarkResults202509.md)** - TileLang compute backend
- **[CUDA C/C++ Backend](./docs/DSLBenchmarkResults202509.md)** - CUDA Native backend
- **[CPU Backend](./docs/DSLBenchmarkResults202509.md)** - CPU backend
