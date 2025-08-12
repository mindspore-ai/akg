[中文版](./README_CN.md)

# AI-driven Kernel Generator (AIKG)

## Table of Contents
- [AI-driven Kernel Generator (AIKG)](#ai-driven-kernel-generator-aikg)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Changelog](#2-changelog)
  - [3. Installation Guide](#3-installation-guide)
  - [4. Configuration](#4-configuration)
    - [4.1 API and Model Configuration](#41-api-and-model-configuration)
    - [4.2 Third-party Dependencies](#42-third-party-dependencies)
    - [4.3 MindSpore 2.7 Frontend Dependencies](#43-mindspore-27-frontend-dependencies)
    - [4.4 Huawei Atlas Inference Series SWFT Backend Dependencies](#44-huawei-atlas-inference-series-swft-backend-dependencies)
    - [4.5 Huawei Atlas A2 Training Series Triton Backend Dependencies](#45-huawei-atlas-a2-training-series-triton-backend-dependencies)
    - [4.6 NVIDIA GPU Triton Backend Dependencies](#46-nvidia-gpu-triton-backend-dependencies)
    - [4.7 Similarity Detection Dependencies](#47-similarity-detection-dependencies)
  - [5. Usage Examples](#5-usage-examples)
  - [6. Design Documentation](#6-design-documentation)
    - [6.1 AIKG General Framework](#61-aikg-general-framework)
    - [6.2 Designer](#62-designer)
    - [6.3 Coder](#63-coder)
    - [6.4 Verifier](#64-verifier)
    - [6.5 Conductor](#65-conductor)
    - [6.6 SWFT Backend](#66-swft-backend)
    - [6.7 Triton Backend](#67-triton-backend)

## 1. Project Overview
AIKG is an AI-driven kernel generator that leverages the code generation capabilities of Large Language Models (LLMs). 
Through LLM-based planning and control of (multi-)agents, AIKG collaboratively accomplishes multi-backend, multi-type AI kernel generation and automatic optimization. 
Additionally, AIKG provides a rich set of submodules for kernel agents, which enables users to build custom agent tasks.

## 2. Changelog
- **CustomDocs**: Support custom reference documents for different Agents to improve generation quality and precision. For detailed configuration instructions, please refer to [Custom Documentation Configuration Guide](./docs/CustomDocs.md)

## 3. Installation Guide
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

## 4. Configuration

### 4.1 API and Model Configuration
AIKG uses environment variables to set the API keys for various Large Language Model (LLM) services. Please configure the appropriate environment variables based on the service you are using:

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434

# Other API interfaces. For detailed supported list, please refer to docs/API.md
export AIKG_XXXXX_API_KEY=xxxxxxxxxxxxxxxxxxx
```
For more information on registering new model configurations in `llm_config.yaml`, orchestrating task workflows in `xxx_config.yaml`, and viewing the current list of supported APIs, please refer to the [API](./docs/API.md) documentation.

### 4.2 Third-party Dependencies
This project uses git submodules to manage certain third-party dependencies.

After initial cloning or pulling updates, please use the following command to initialize and download `aikg`-related dependencies:
```bash
# Initialize and pull aikg-related submodules
git submodule update --init --remote "aikg/thirdparty/*"
```

### 4.3 MindSpore 2.7 Frontend Dependencies
Supported Python versions: Python 3.11, Python 3.10, Python 3.9
Supported system versions: aarch64, x86_64
```
# Example installation package for python3.11 + aarch64
pip install https://repo.mindspore.cn/mindspore/mindspore/version/202506/20250619/master_20250619160020_1261ff4ce06d6f2dc4ce446139948a3e4e9c966b_newest/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl
```

### 4.4 Huawei Atlas Inference Series SWFT Backend Dependencies
Please refer to: https://gitee.com/mindspore/akg/swft

### 4.5 Huawei Atlas A2 Training Series Triton Backend Dependencies
Please refer to: https://gitee.com/ascend/triton-ascend

### 4.6 NVIDIA GPU Triton Backend Dependencies
Please refer to: https://github.com/triton-lang/triton


### 4.7 Similarity Detection Dependencies
The text similarity detection tool text2vec-large-chinese: If the model cannot be loaded automatically, manually download it to the thirdparty directory.
After downloading the model, add its local path to the corresponding YAML configuration in the database. For detailed configuration instructions, please refer to the [DataBase](./docs/DataBase.md) documentation.
```bash
bash download.sh --with_local_model
```

## 5. Usage Examples
For a simplified workflow demonstrating AIKG's automatic kernel generation capabilities, please refer to the [Tutorial](./docs/Tutorial.md) documentation and example code in the `examples` directory.

## 6. Design Documentation
### 6.1 AIKG General Framework
- `Task`: Please refer to [Task](./docs/Task.md) documentation
- `Trace`: Please refer to [Trace](./docs/Trace.md) documentation
- `TaskPool`: Please refer to [TaskPool](./docs/TaskPool.md) documentation
- `DevicePool`: Please refer to [DevicePool](./docs/DevicePool.md) documentation
- `Database`: Please refer to [DataBase](./docs/Database.md) documentation

### 6.2 Designer
Please refer to [Designer](./docs/Designer.md) documentation

### 6.3 Coder
Please refer to [Coder](./docs/Coder.md) documentation

### 6.4 Verifier
Please refer to [Verifier](./docs/Verifier.md) documentation

### 6.5 Conductor
Please refer to [Conductor](./docs/Conductor.md) documentation

### 6.6 SWFT Backend
Please refer to [SWFT](./docs/SWFT.md) documentation

### 6.7 Triton Backend
Please refer to [Triton](./docs/Triton.md) documentation
