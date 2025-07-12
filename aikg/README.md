[中文版](./README_CN.md)

# AI-driven Kernel Generator (AIKG)

## Table of Contents
- [AI-driven Kernel Generator (AIKG)](#ai-driven-kernel-generator-aikg)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Installation Guide](#2-installation-guide)
  - [3. Configuration](#3-configuration)
    - [3.1 API and Model Configuration](#31-api-and-model-configuration)
    - [3.2 Third-party Dependencies](#32-third-party-dependencies)
    - [3.3 MindSpore 2.7 Frontend Dependencies](#33-mindspore-27-frontend-dependencies)
    - [3.4 Huawei Atlas Inference Series SWFT Backend Dependencies](#34-huawei-atlas-inference-series-swft-backend-dependencies)
    - [3.5 Huawei Atlas A2 Training Series Triton Backend Dependencies](#35-huawei-atlas-a2-training-series-triton-backend-dependencies)
    - [3.6 NVIDIA GPU Triton Backend Dependencies](#36-nvidia-gpu-triton-backend-dependencies)
  - [4. Usage Examples](#4-usage-examples)
  - [5. Design Documentation](#5-design-documentation)
    - [5.1 AIKG General Framework](#51-aikg-general-framework)
    - [5.2 Designer](#52-designer)
    - [5.3 Coder](#53-coder)
    - [5.4 Verifier](#54-verifier)
    - [5.5 Conductor](#55-conductor)
    - [5.6 SWFT Backend](#56-swft-backend)
    - [5.7 Triton Backend](#57-triton-backend)

## 1. Project Overview
AIKG is an AI-driven kernel generator that leverages the code generation capabilities of Large Language Models (LLMs). 
Through LLM-based planning and control of (multi-)agents, AIKG collaboratively accomplishes multi-backend, multi-type AI kernel generation and automatic optimization. 
Additionally, AIKG provides a rich set of submodules for kernel agents, which enables users to build custom agent tasks.

## 2. Installation Guide
```bash
# Create conda environment (optional, recommended Python 3.9/3.10/3.11)
conda create -n aikg python=3.11
conda activate aikg

# Or create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate

# Install dependencies via pip
pip install -r requirements.txt

# Setup & install
bash build.sh
pip install output/ai_kernel_generator-*-py3-none-any.whl
```

## 3. Configuration

### 3.1 API and Model Configuration
AIKG uses environment variables to set the API keys and service endpoints for various Large Language Model (LLM) services. Please configure the appropriate environment variables based on the service you are using:

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434

# SiliconFlow (https://www.siliconflow.cn/)
export AIKG_SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# DeepSeek (https://www.deepseek.com/)
export AIKG_DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# Volcengine (https://www.volcengine.com/)
export AIKG_HUOSHAN_API_KEY=0cbf8bxxxxxx

# Moonshot (https://www.moonshot.cn/)
export AIKG_MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxxxxx
```
For more detailed information on how to configure and use `llm_config.yaml` (for registering new models) and `xxx_config.yaml` (for orchestrating task workflows), please refer to the comprehensive [API](./docs/API.md) documentation.

### 3.2 Third-party Dependencies
This project uses git submodules to manage certain third-party dependencies.

After initial cloning or pulling updates, please use the following command to initialize and download `aikg`-related dependencies:
```bash
# Initialize and pull aikg-related submodules
git submodule update --init --remote "aikg/thirdparty/*"
```

### 3.3 MindSpore 2.7 Frontend Dependencies
Supported Python versions: Python 3.11, Python 3.10, Python 3.9
Supported system versions: aarch64, x86_64
```
# Example installation package for python3.11 + aarch64
pip install https://repo.mindspore.cn/mindspore/mindspore/version/202506/20250619/master_20250619160020_1261ff4ce06d6f2dc4ce446139948a3e4e9c966b_newest/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl
```

### 3.4 Huawei Atlas Inference Series SWFT Backend Dependencies
Please refer to: https://gitee.com/mindspore/akg/swft

### 3.5 Huawei Atlas A2 Training Series Triton Backend Dependencies
Please refer to: https://gitee.com/ascend/triton-ascend

### 3.6 NVIDIA GPU Triton Backend Dependencies
Please refer to: https://github.com/triton-lang/triton

## 4. Usage Examples
For a simplified workflow demonstrating AIKG's automatic kernel generation capabilities, please refer to the [Tutorial](./docs/Tutorial.md) documentation and example code in the `examples` directory.

## 5. Design Documentation
### 5.1 AIKG General Framework
- `Task`: Please refer to [Task](./docs/Task.md) documentation
- `Trace`: Please refer to [Trace](./docs/Trace.md) documentation
- `TaskPool`: Please refer to [TaskPool](./docs/TaskPool.md) documentation
- `DevicePool`: Please refer to [DevicePool](./docs/DevicePool.md) documentation

### 5.2 Designer
Please refer to [Designer](./docs/Designer.md) documentation

### 5.3 Coder
Please refer to [Coder](./docs/Coder.md) documentation

### 5.4 Verifier
Please refer to [Verifier](./docs/Verifier.md) documentation

### 5.5 Conductor
Please refer to [Conductor](./docs/Conductor.md) documentation

### 5.6 SWFT Backend
Please refer to [SWFT](./docs/SWFT.md) documentation

### 5.7 Triton Backend
Please refer to [Triton](./docs/Triton.md) documentation 
