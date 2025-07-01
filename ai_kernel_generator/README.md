# AI-driven Kernel Generator(AIKG)

## 目录
- [AI-driven Kernel Generator(AIKG)](#ai-driven-kernel-generatoraikg)
  - [目录](#目录)
  - [1. 项目简介](#1-项目简介)
  - [2. 安装流程](#2-安装流程)
  - [3. 配置](#3-配置)
    - [3.1 配置API环境变量（可选）](#31-配置api环境变量可选)
    - [3.2 MindSpore 2.7版本 前端依赖](#32-mindspore-27版本-前端依赖)
    - [3.3 华为Atlas推理系列产品 SWFT 后端依赖](#33-华为atlas推理系列产品-swft-后端依赖)
    - [3.4 华为Atlas A2训练系列产品 Triton 后端依赖](#34-华为atlas-a2训练系列产品-triton-后端依赖)
    - [3.5 NVIDIA GPU Triton 后端依赖](#35-nvidia-gpu-triton-后端依赖)
  - [4. 运行示例](#4-运行示例)
  - [5. 适配新模型](#5-适配新模型)
    - [5.1 通用参数](#51-通用参数)
    - [5.2 全流程LLM配置 \& 通用设置](#52-全流程llm配置--通用设置)
  - [6. 设计文档](#6-设计文档)
    - [6.1 AIKG通用框架](#61-aikg通用框架)
    - [6.2 Designer](#62-designer)
    - [6.3 Coder](#63-coder)
    - [6.4 Verifier](#64-verifier)
    - [6.5 SWFT Backend](#65-swft-backend)
    - [6.6 Triton Backend](#66-triton-backend)
  - [7. 版本说明](#7-版本说明)
  - [8. 许可证](#8-许可证)

## 1. 项目简介
AIKG 是一款 AI 驱动的算子生成器。
AIKG 基于大语言模型(LLM)的代码生成能力，通过大语言模型规划和控制多个 Agent，协同完成多后端、多类型的AI算子生成和自动优化。


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

### 3.1 配置API环境变量（可选）
```
export AIKG_VLLM_API_BASE=http://localhost:8000/v1 # 本地或远程VLLM服务器地址
export AIKG_OLLAMA_API_BASE=http://localhost:11434 # 本地或远程Ollama服务器地址
export AIKG_SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxx # 硅流key
export AIKG_DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxx # DeepSeek key
export AIKG_HUOSHAN_API_KEY=0cbf8bxxxxxx # 火山key
```

### 3.2 MindSpore 2.7版本 前端依赖
支持python版本：python3.11、python3.10、python3.9
支持系统版本：aarch64、x86_64
```
# python3.11 + aarch64的安装包示例
pip install https://repo.mindspore.cn/mindspore/mindspore/version/202506/20250619/master_20250619160020_1261ff4ce06d6f2dc4ce446139948a3e4e9c966b_newest/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl
```

### 3.3 华为Atlas推理系列产品 SWFT 后端依赖
请参考：https://gitee.com/mindspore/akg/swft

### 3.4 华为Atlas A2训练系列产品 Triton 后端依赖
请参考：https://gitee.com/ascend/triton-ascend


### 3.5 NVIDIA GPU Triton 后端依赖
请参考：https://github.com/triton-lang/triton


## 4. 运行示例
通过AIKG完成算子自动生成的简易流程，请参考[Tutorial](./docs/Tutorial.md)文档以及`examples`目录中示例代码。


## 5. 适配新模型
在 `ai_kernel_generator/core/llm/llm_config.yaml` 文件中可以配置新的模型。每个模型配置可包含以下参数：

### 5.1 通用参数
- `api_base`: API 基础 URL
- `model`: 模型名称
- `max_tokens`: 最大生成 token 数
- `temperature`: 温度参数，控制随机性
- `top_p`: 核采样参数，控制多样性
- `frequency_penalty`: 频率惩罚，控制重复
- `presence_penalty`: 存在惩罚，控制主题重复

配置完成后，可以通过 `create_model("my_model_name")` 来使用新配置的模型。

### 5.2 全流程LLM配置 & 通用设置
在aikg完整流程里，可以通过设置自定义`config.yaml`的方式来控制每个子任务调用的LLM
```python
config = load_config() # 调用默认配置 default_config.yaml
config = load_config("/your-path-to-config/vllm_deepseek_r1_config.yaml")
task = Task(
    ...
    config=config,
)
```


## 6. 设计文档
### 6.1 AIKG通用框架
- `Task`: 请参考 [Task](./docs/Task.md) 文档
- `Trace`: 请参考 [Trace](./docs/Trace.md) 文档
- `TaskPool`: 请参考 [TaskPool](./docs/TaskPool.md) 文档
- `DevicePool`: 请参考 [DevicePool](./docs/DevicePool.md) 文档

### 6.2 Designer
请参考 [Designer](./docs/Designer.md) 文档

### 6.3 Coder
请参考 [Coder](./docs/Coder.md) 文档

### 6.4 Verifier
请参考 [Verifier](./docs/Verifier.md) 文档

### 6.5 SWFT Backend
请参考 [SWFT](./docs/SWFT.md) 文档

### 6.6 Triton Backend
请参考 [Triton](./docs/Triton.md) 文档

## 7. 版本说明

版本说明详见[RELEASE](../RELEASE.md)。

## 8. 许可证

[Apache License 2.0](LICENSE)。