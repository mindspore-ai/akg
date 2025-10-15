[中文版](./README_CN.md)

<div align="center">
  <img src="./aikg_logo.jpg" alt="AIKG Logo" width="400">
</div>

<div align="center">

# AI-driven Kernel Generator (AIKG)

</div>

<details>
<summary><b>📋 Table of Contents</b></summary>

- [AI-driven Kernel Generator (AIKG)](#ai-driven-kernel-generator-aikg)
  - [📘 1. Project Overview](#-1-project-overview)
  - [🗓️ 2. Changelog](#️-2-changelog)
  - [🛠️ 3. Installation \& Deployment Guide](#️-3-installation--deployment-guide)
  - [⚙️ 4. Configuration](#️-4-configuration)
    - [Configuration Quick Guide](#configuration-quick-guide)
      - [Step 1: Basic Environment Configuration](#step-1-basic-environment-configuration)
        - [API and Model Configuration](#api-and-model-configuration)
        - [Third-party Dependencies](#third-party-dependencies)
      - [Step 2: Frontend Dependencies Configuration](#step-2-frontend-dependencies-configuration)
        - [MindSpore 2.7 Frontend Dependencies(Optional)](#mindspore-27-frontend-dependenciesoptional)
      - [Step 3: Backend Dependencies Configuration](#step-3-backend-dependencies-configuration)
      - [Step 4: Optional Tools Configuration](#step-4-optional-tools-configuration)
        - [Similarity Detection Dependencies](#similarity-detection-dependencies)
  - [▶️ 5. Tutorial Examples](#️-5-tutorial-examples)
  - [📐 6. Design Documentation](#-6-design-documentation)
    - [Core Framework](#core-framework)
    - [Core Components](#core-components)
    - [Backend Support](#backend-support)

</details>

## 📘 1. Project Overview
AIKG is an AI-driven kernel generator that leverages the code generation capabilities of Large Language Models (LLMs). 
Through LLM-based planning and control of (multi-)agents, AIKG collaboratively accomplishes multi-backend, multi-type AI kernel generation and automatic optimization. 
Additionally, AIKG provides a rich set of submodules for kernel agents, which enables users to build custom agent tasks.

## 🗓️ 2. Changelog
- 2025-10-14: Added TileLang_CUDA DSL support. See [Benchmark Results](./docs/DSLBenchmarkResults202509.md) for KernelBench Level1 success rates.
- 2025-09-26: Added CUDA C and C++ DSL support. See [Benchmark Results](./docs/DSLBenchmarkResults202509.md) for KernelBench Level1 success rates.
- 2025-09-14: KernelBench Level1 kernel generation success rate updated, see [Benchmark Results](./docs/BenchmarkResults202509.md).
- 2025-08-12: Introduced Doc-Driven Integration; by following a unified documentation specification, you can quickly and flexibly integrate new DSLs/frontends/backends (see [Doc-Driven Integration Guide](./docs/DocDrivenIntegration.md)).
- 2025-06-27: Initial AIKG release with code generation support for Triton and SWFT backends.

## 🛠️ 3. Installation & Deployment Guide
```bash
# 1. Environment Setup
# 1.1 Create conda environment (optional, recommended Python 3.9/3.10/3.11)
conda create -n aikg python=3.11
conda activate aikg

# 1.2 Or create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies via pip
pip install -r requirements.txt

# 3. whl installation / environment setup
# 3.1 Install from whl
bash build.sh
pip install output/ai_kernel_generator-*-py3-none-any.whl

# 3.2 Or setup environment variables
cd aikg
source env.sh
```

## ⚙️ 4. Configuration

### Configuration Quick Guide

#### Step 1: Basic Environment Configuration

##### API and Model Configuration
AIKG uses environment variables to set the API keys for various Large Language Model (LLM) services. Please configure the appropriate environment variables based on the service you are using:

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Other API interfaces. For detailed supported list, please refer to docs/API.md
export AIKG_XXX_API_KEY=xxx

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434
```
Additional configuration options:
- **Task Orchestration Plan Configuration**: Declares a task's complete runtime scheme (including `agent_model_config`, `workflow_config_path`, `docs_dir`, etc.). Common plan files: `default_triton_config.yaml`, `vllm_triton_coderonly_config.yaml`. See [Task Orchestration Plan Configuration](./docs/TaskOrchestrationPlan.md).
- **Model Configuration**: `llm_config.yaml` contains preset configurations for various LLM providers (DeepSeek, Qwen, Moonshot, etc.). The `agent_model_config` in the plan references presets from this file.
- **Workflow Definition**: Specify the workflow YAML via `workflow_config_path` to define agent execution order and constraints (e.g., `default_workflow.yaml`, `coder_only_workflow.yaml`). See [Workflow System Design Document](./docs/Workflow.md).
- **Doc-Driven Integration**: Provide reference docs for agents via the plan's `docs_dir`. See [Doc-Driven Integration Guide](./docs/DocDrivenIntegration.md).

For detailed configuration instructions, please refer to [API Configuration Documentation](./docs/API.md).

##### Third-party Dependencies
This project uses git submodules to manage certain third-party dependencies.

After initial cloning or pulling updates, please use the following command to initialize and download `aikg`-related dependencies:
```bash
# Initialize and pull aikg-related submodules
git submodule update --init "aikg/thirdparty/*"
```

#### Step 2: Frontend Dependencies Configuration

##### MindSpore 2.7 Frontend Dependencies(Optional)
Supported Python versions: 3.11, 3.10, 3.9
Supported system architectures: aarch64, x86_64
Prefer the official installation guide to choose environment and method: [MindSpore 2.7 Installation Guide](https://www.mindspore.cn/en/install)

#### Step 3: Backend Dependencies Configuration
Choose the appropriate backend based on your hardware platform:

| Platform | Backend | Reference Link |
|----------|---------|----------------|
| Huawei Atlas A2 Training Series | Triton | https://gitee.com/ascend/triton-ascend |
| NVIDIA GPU | Triton | https://github.com/triton-lang/triton |
| Huawei Atlas Inference Series | SWFT | https://gitee.com/mindspore/akg/tree/br_aikg/swft |

#### Step 4: Optional Tools Configuration

##### Similarity Detection Dependencies
The text similarity detection tool text2vec-large-chinese: If the model cannot be loaded automatically, manually download it to the thirdparty directory.
After downloading the model, add its local path to the corresponding YAML configuration in the database. For detailed configuration instructions, please refer to the [DataBase](./docs/DataBase.md) documentation.
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
| `run_mindspore_triton_single.py` | Single operator example (MindSpore + Triton, Ascend 910B4). |
| `run_mindspore_triton_parallel.py` | Parallel multi-operator example (MindSpore + Triton, Ascend 910B4). |
| `run_numpy_swft_relu.py` | SWFT ReLU example (Ascend 310P3). |
| `run_numpy_swft_swiglu.py` | SWFT SwiGLU example (Ascend 310P3). |

For more getting started steps and parameter notes, please refer to the [Tutorial](./docs/Tutorial.md).

## 📐 6. Design Documentation

> We recommend reading the [Task Orchestration Plan Configuration](./docs/TaskOrchestrationPlan.md) first for the overall task plan and entry points; workflow details are in [Workflow](./docs/Workflow.md) and documentation specs are in [Doc-Driven Integration](./docs/DocDrivenIntegration.md).

### Core Framework
- **[Task](./docs/Task.md)** - Task management module
- **[Trace](./docs/Trace.md)** - Execution tracking module  
- **[TaskPool](./docs/TaskPool.md)** - Task pool management
- **[DevicePool](./docs/DevicePool.md)** - Device pool management
- **[DataBase](./docs/DataBase.md)** - Database module

### Core Components
- **[Designer](./docs/Designer.md)** - Kernel designer
- **[Coder](./docs/Coder.md)** - Code generator
- **[Verifier](./docs/Verifier.md)** - Verifier
- **[Conductor](./docs/Conductor.md)** - Task orchestrator

### Backend Support
- **[SWFT Backend](./docs/SWFT.md)** - Huawei Atlas inference series backend
- **[Triton Backend](./docs/Triton.md)** - Triton compute backend
