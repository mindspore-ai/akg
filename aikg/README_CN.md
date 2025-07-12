[English Version](./README.md)

# AI-driven Kernel Generator(AIKG)

## 目录
- [AI-driven Kernel Generator(AIKG)](#ai-driven-kernel-generatoraikg)
  - [目录](#目录)
  - [1. 项目简介](#1-项目简介)
  - [2. 安装流程](#2-安装流程)
  - [3. 配置](#3-配置)
    - [3.1 API与模型配置](#31-api与模型配置)
    - [3.2 第三方依赖](#32-第三方依赖)
    - [3.3 MindSpore 2.7版本 前端依赖](#33-mindspore-27版本-前端依赖)
    - [3.4 华为Atlas推理系列产品 SWFT 后端依赖](#34-华为atlas推理系列产品-swft-后端依赖)
    - [3.5 华为Atlas A2训练系列产品 Triton 后端依赖](#35-华为atlas-a2训练系列产品-triton-后端依赖)
    - [3.6 NVIDIA GPU Triton 后端依赖](#36-nvidia-gpu-triton-后端依赖)
  - [4. 运行示例](#4-运行示例)
  - [5. 设计文档](#5-设计文档)
    - [5.1 AIKG通用框架](#51-aikg通用框架)
    - [5.2 Designer](#52-designer)
    - [5.3 Coder](#53-coder)
    - [5.4 Verifier](#54-verifier)
    - [5.5 Conductor](#55-conductor)
    - [5.6 SWFT Backend](#56-swft-backend)
    - [5.7 Triton Backend](#57-triton-backend)

## 1. 项目简介
AIKG 是一款 AI 驱动的算子生成器。
AIKG 利用大语言模型(LLM)的代码生成能力，通过大语言模型规划和控制（多个）Agent 协同完成多后端、多类型的AI算子生成和自动优化。
同时 AIKG 提供丰富的算子Agent相关子模块，用户可组合构建自定义算子Agents任务。


## 2. 安装流程
```bash
# 使用conda环境（可选， 推荐python3.9/3.10/3.11版本）
conda create -n aikg python=3.11
conda activate aikg

# 或者创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/active

# pip安装依赖
pip install -r requirements.txt

# setup & install
bash build.sh
pip install output/ai_kernel_generator-*-py3-none-any.whl
```


## 3. 配置

### 3.1 API与模型配置
AIKG 通过环境变量来设置不同大语言模型（LLM）服务的 API Key 和服务地址（Endpoint）。请根据您使用的服务，配置相应的环境变量：

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434

# 硅基流动 (https://www.siliconflow.cn/)
export AIKG_SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# DeepSeek (https://www.deepseek.com/)
export AIKG_DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# 火山引擎 (https://www.volcengine.com/)
export AIKG_HUOSHAN_API_KEY=0cbf8bxxxxxx

# Moonshot (https://www.moonshot.cn/)
export AIKG_MOONSHOT_API_KEY=sk-xxxxxxxxxxxxxxxxxxx
```
关于如何配置和使用 `llm_config.yaml`（用于注册新模型）和 `xxx_config.yaml`（用于编排任务流程）的更详细信息，请参考详细的 [API](./docs/CN/API.md) 文档。

### 3.2 第三方依赖
本项目使用 git submodule 管理部分第三方依赖。

初次克隆或拉取更新后，请使用以下命令初始化并下载 `aikg` 相关的依赖：
```bash
# 初始化并拉取 aikg 相关的子模块
git submodule update --init --remote "aikg/thirdparty/*"
```

### 3.3 MindSpore 2.7版本 前端依赖
支持python版本：python3.11、python3.10、python3.9
支持系统版本：aarch64、x86_64
```
# python3.11 + aarch64的安装包示例
pip install https://repo.mindspore.cn/mindspore/mindspore/version/202506/20250619/master_20250619160020_1261ff4ce06d6f2dc4ce446139948a3e4e9c966b_newest/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl
```

### 3.4 华为Atlas推理系列产品 SWFT 后端依赖
请参考：https://gitee.com/mindspore/akg/swft

### 3.5 华为Atlas A2训练系列产品 Triton 后端依赖
请参考：https://gitee.com/ascend/triton-ascend


### 3.6 NVIDIA GPU Triton 后端依赖
请参考：https://github.com/triton-lang/triton


## 4. 运行示例
通过AIKG完成算子自动生成的简易流程，请参考[Tutorial](./docs/CN/Tutorial.md)文档以及`examples`目录中示例代码。


## 5. 设计文档
### 5.1 AIKG通用框架
- `Task`: 请参考 [Task](./docs/CN/Task.md) 文档
- `Trace`: 请参考 [Trace](./docs/CN/Trace.md) 文档
- `TaskPool`: 请参考 [TaskPool](./docs/CN/TaskPool.md) 文档
- `DevicePool`: 请参考 [DevicePool](./docs/CN/DevicePool.md) 文档

### 5.2 Designer
请参考 [Designer](./docs/CN/Designer.md) 文档

### 5.3 Coder
请参考 [Coder](./docs/CN/Coder.md) 文档

### 5.4 Verifier
请参考 [Verifier](./docs/CN/Verifier.md) 文档

### 5.5 Conductor
请参考 [Conductor](./docs/CN/Conductor.md) 文档

### 5.6 SWFT Backend
请参考 [SWFT](./docs/CN/SWFT.md) 文档

### 5.7 Triton Backend
请参考 [Triton](./docs/CN/Triton.md) 文档
