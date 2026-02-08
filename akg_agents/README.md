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
    - [API Configuration Example](#api-configuration-example)
    - [Launch AKG\_CLI](#launch-akg_cli)
    - [Usage](#usage)
  - [⚙️ 4. Configuration](#️-4-configuration)
    - [Configuration Quick Guide](#configuration-quick-guide)
      - [Step 1: Basic Environment Configuration](#step-1-basic-environment-configuration)
        - [API and Model Configuration](#api-and-model-configuration)
        - [Third-party Dependencies](#third-party-dependencies)
      - [Step 2: Backend Dependencies Configuration](#step-2-backend-dependencies-configuration)
      - [Step 3: Optional Tools Configuration](#step-3-optional-tools-configuration)
        - [Similarity Detection Dependencies (RAG-related)](#similarity-detection-dependencies-rag-related)
  - [▶️ 5. Tutorial Examples](#️-5-tutorial-examples)
  - [📐 6. Design Documentation](#-6-design-documentation)
    - [Core Framework](#core-framework)
    - [Core Components](#core-components)
    - [Service Architecture](#service-architecture)
    - [Backend Support](#backend-support)

</details>

## 📘 1. Project Overview
AIKG is an AI-driven kernel generator.
AIKG leverages the code generation capabilities of Large Language Models (LLMs) to collaboratively accomplish multi-backend, multi-type AI kernel generation and automatic optimization through LLM-based planning and control of multi-agents.
Additionally, AIKG provides a rich set of submodules for kernel agents, enabling users to combine and build custom agent tasks.

<div align="center" style="background-color:white">
  <img src="./akg_agents.png" alt="AIKG Architecture" width="600">
</div>

## 🗓️ 2. Changelog
- 2025-12-01: Introduced LangGraph refactoring for task orchestration. New `LangGraphTask` replaces original `Task Orchestration` scheme. Supports Python-defined workflows, graph visualization, and type-safe state management. API fully compatible with original `Task`. See [LangGraph Documentation](./docs/LangGraph.md).
- 2025-11-25: Supported service architecture, including `client-server-worker` separation architecture, supporting various flexible concurrency requirements. See [Service Architecture Documentation](./docs/ServerArchitecture.md).
- 2025-10-14: Supported TileLang_CUDA backend code generation capability. See [Benchmark Results](./docs/DSLBenchmarkResults202509.md) for KernelBench Level1 TileLang_CUDA backend kernel generation success rates.
- 2025-09-26: Supported CUDA C and CPP backend code generation capability. See [Benchmark Results](./docs/DSLBenchmarkResults202509.md) for KernelBench Level1 CUDA C and CPP backend kernel generation success rates.
- 2025-09-14: Updated KernelBench Level1 kernel generation success rate. See [Benchmark Results](./docs/BenchmarkResults202509.md).
- 2025-08-12: Supported "Doc-Driven Integration"; by following a unified documentation specification, you can quickly and flexibly integrate new DSLs/frontends/backends (see [Doc-Driven Integration Guide](./docs/DocDrivenIntegration.md)).
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
# pip install -r akg_agents/rag_requirements.txt  # Optional: if RAG features are needed

# 4. Install AIKG
pip install -e ./akg_agents

# 5. Setup environment variables
cd ./akg_agents
source env.sh
```

### API Configuration Example
```bash
# LLM API (Using DeepSeek as an example; can be replaced with other supported model providers, see docs/API.md)
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"
export AKG_AGENTS_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"  # or "disabled"

# Embedding API (Optional, for RAG retrieval)
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="Qwen/Qwen3-Embedding-8B"
export AKG_AGENTS_EMBEDDING_API_KEY="YOUR_API_KEY"
```

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
AIKG uses environment variables to set the API keys for various Large Language Model (LLM) services. Please configure the appropriate environment variables based on the service you are using:

```bash
# API interfaces for various vendors. For detailed supported list, please refer to docs/API.md
export AKG_AGENTS_XXX_API_KEY=xxx

# VLLM (https://github.com/vllm-project/vllm)
export AKG_AGENTS_VLLM_API_BASE=http://localhost:8000/v1

...
```

More configuration options:
- **LangGraph Workflow Configuration**: Uses LangGraph to define task execution flows, supporting Python code defined graph structures, state management, and visualization. See [LangGraph Documentation](./docs/LangGraph.md).
  > Note: The original **Task Orchestration Plan Configuration** is temporarily compatible. See [Task Orchestration Plan Configuration](./docs/TaskOrchestrationPlan.md).
- **Model Configuration**: `llm_config.yaml` contains preset configurations for various LLM providers (DeepSeek, Qwen, Moonshot, etc.).
- **Doc-Driven Integration**: Provide reference documentation directories for agents via `docs_dir`. See [Doc-Driven Integration Guide](./docs/DocDrivenIntegration.md).

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
| Huawei Atlas Inference Series | SWFT | https://gitee.com/mindspore/akg_agents/tree/br_akg_agents/swft |
| NVIDIA GPU | TileLang | https://github.com/tile-ai/tilelang |
| Huawei Atlas A2 Training Series | TileLang | https://github.com/tile-ai/tilelang |
| NVIDIA GPU | CUDA C/C++ | https://docs.nvidia.com/cuda/ |

#### Step 3: Optional Tools Configuration

##### Similarity Detection Dependencies (RAG-related)
Text sentence similarity detection tool text2vec-large-chinese: If the model cannot be loaded automatically, you need to manually download it to the thirdparty directory.
Add the downloaded model address to the corresponding yaml in the database, please refer to [DataBase](./docs/DataBase.md) documentation.
```bash
bash download.sh --with_local_model
```

> 💡 **Configuration Tips**: 
> - For detailed API configuration, please refer to [API Documentation](./docs/API.md) 
> - For database configuration, please refer to [DataBase Documentation](./docs/DataBase.md)
> - For more configuration options, please refer to the dedicated documentation for each component

## ▶️ 5. Tutorial Examples

Below are common examples in the `examples/` directory:

| Example | Description |
|--------|-------------|
| `run_torch_npu_triton_single.py` | Single operator example (Torch + Triton, Ascend). |
| `run_torch_evolve_triton.py` | Evolutionary algorithm operator optimization example (Torch + Triton). |
| `run_numpy_swft_relu.py` | SWFT ReLU example (Ascend 310P3). |
| `run_numpy_swft_swiglu.py` | SWFT SwiGLU example (Ascend 310P3). |
| `run_cuda_to_ascend_conversion.py` | CUDA to Ascend operator conversion example. |
| `run_client_server_worker.py` | Client-Server distributed run example. |
| `kernel_profile.py` | Operator performance Profiling example. |
| `handwrite_optimization_analyzer.py` | Handwrite optimization analyzer example. |

For more getting started steps and parameter notes, please refer to the [Tutorial](./docs/Tutorial.md).


## 📐 6. Design Documentation

> It is recommended to read [LangGraph Documentation](./docs/LangGraph.md) first to understand the latest task orchestration scheme; for workflow details see [Workflow](./docs/Workflow.md), and for documentation specifications see [Doc-Driven Integration Guide](./docs/DocDrivenIntegration.md).

### Core Framework
- **[LangGraph Task](./docs/LangGraph.md)** - Task management module (LangGraph version)
- **[Trace](./docs/Trace.md)** - Execution tracking module  
- **[TaskPool](./docs/TaskPool.md)** - Task pool management
- **[DevicePool](./docs/DevicePool.md)** - Device pool management
- **[DataBase](./docs/DataBase.md)** - Database module

### Core Components
- **[Designer](./docs/Designer.md)** - Operator designer
- **[Coder](./docs/Coder.md)** - Code generator
- **[Verifier](./docs/Verifier.md)** - Verifier
- **[Conductor](./docs/Conductor.md)** - Task orchestrator

### Service Architecture
- **[Server Architecture](./docs/ServerArchitecture.md)** - Service architecture documentation, including Client-Server-Worker architecture, WorkerManager load balancing, convenience function usage, etc.

### Backend Support
- **[Triton Backend (Ascend/CUDA)](./docs/Triton.md)** - Triton compute backend
- **[TileLang Backend (Ascend/CUDA)](./docs/DSLBenchmarkResults202509.md)** - TileLang compute backend
- **[CUDA C/C++ Backend](./docs/DSLBenchmarkResults202509.md)** - CUDA Native backend
- **[SWFT Backend](./docs/SWFT.md)** - Huawei Atlas inference series backend
- **[CPU Backend](./docs/DSLBenchmarkResults202509.md)** - CPU backend
