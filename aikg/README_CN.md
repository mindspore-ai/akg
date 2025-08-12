[English Version](./README.md)

# AI-driven Kernel Generator(AIKG)

## 目录
- [AI-driven Kernel Generator(AIKG)](#ai-driven-kernel-generatoraikg)
  - [目录](#目录)
  - [1. 项目简介](#1-项目简介)
  - [2. 更新日志](#2-更新日志)
  - [3. 安装流程](#3-安装流程)
  - [4. 配置](#4-配置)
    - [4.1 API与模型配置](#41-api与模型配置)
    - [4.2 第三方依赖](#42-第三方依赖)
    - [4.3 MindSpore 2.7版本 前端依赖](#43-mindspore-27版本-前端依赖)
    - [4.4 华为Atlas推理系列产品 SWFT 后端依赖](#44-华为atlas推理系列产品-swft-后端依赖)
    - [4.5 华为Atlas A2训练系列产品 Triton 后端依赖](#45-华为atlas-a2训练系列产品-triton-后端依赖)
    - [4.6 NVIDIA GPU Triton 后端依赖](#46-nvidia-gpu-triton-后端依赖)
    - [4.7 相似性检测依赖](#47-相似性检测依赖)
  - [5. 运行示例](#5-运行示例)
  - [6. 设计文档](#6-设计文档)
    - [6.1 AIKG通用框架](#61-aikg通用框架)
    - [6.2 Designer](#62-designer)
    - [6.3 Coder](#63-coder)
    - [6.4 Verifier](#64-verifier)
    - [6.5 Conductor](#65-conductor)
    - [6.6 SWFT Backend](#66-swft-backend)
    - [6.7 Triton Backend](#67-triton-backend)

## 1. 项目简介
AIKG 是一款 AI 驱动的算子生成器。
AIKG 利用大语言模型(LLM)的代码生成能力，通过大语言模型规划和控制（多个）Agent 协同完成多后端、多类型的AI算子生成和自动优化。
同时 AIKG 提供丰富的算子Agent相关子模块，用户可组合构建自定义算子Agents任务。

## 2. 更新日志
- **CustomDocs**: 支持为不同Agent自定义参考文档，提升生成质量和精度。详细配置说明请参考 [自定义文档配置指南](./docs/CN/CustomDocs.md)

## 3. 安装流程
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


## 4. 配置

### 4.1 API与模型配置
AIKG 通过环境变量来设置不同大语言模型（LLM）服务的 API。请根据您使用的服务，配置相应的环境变量：

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434

# 其他API接口。详细支持列表请参考docs/API.md
export AIKG_XXXXX_API_KEY=xxxxxxxxxxxxxxxxxxx
```
关于注册新模型配置 `llm_config.yaml`、 编排任务流程设置 `xxx_config.yaml`、查看当前支持API列表 等更多信息，请参考 [API](./docs/CN/API.md) 文档。

### 4.2 第三方依赖
本项目使用 git submodule 管理部分第三方依赖。

初次克隆或拉取更新后，请使用以下命令初始化并下载 `aikg` 相关的依赖：
```bash
# 初始化并拉取 aikg 相关的子模块
git submodule update --init --remote "aikg/thirdparty/*"
```

### 4.3 MindSpore 2.7版本 前端依赖
支持python版本：python3.11、python3.10、python3.9
支持系统版本：aarch64、x86_64
```
# python3.11 + aarch64的安装包示例
pip install https://repo.mindspore.cn/mindspore/mindspore/version/202506/20250619/master_20250619160020_1261ff4ce06d6f2dc4ce446139948a3e4e9c966b_newest/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl
```

### 4.4 华为Atlas推理系列产品 SWFT 后端依赖
请参考：https://gitee.com/mindspore/akg/swft

### 4.5 华为Atlas A2训练系列产品 Triton 后端依赖
请参考：https://gitee.com/ascend/triton-ascend


### 4.6 NVIDIA GPU Triton 后端依赖
请参考：https://github.com/triton-lang/triton


### 4.7 相似性检测依赖
句子相似性检测工具text2vec-large-chinese： 若无法自动加载模型，需要手动下载到thirdparty目录下
将下载后的模型地址添加到database对应的yaml中，请参考  [DataBase](./docs/CN/DataBase.md) 文档
```bash
bash download.sh --with_local_model
```


## 5. 运行示例
通过AIKG完成算子自动生成的简易流程，请参考[Tutorial](./docs/CN/Tutorial.md)文档以及`examples`目录中示例代码。


## 6. 设计文档
### 6.1 AIKG通用框架
- `Task`: 请参考 [Task](./docs/CN/Task.md) 文档
- `Trace`: 请参考 [Trace](./docs/CN/Trace.md) 文档
- `TaskPool`: 请参考 [TaskPool](./docs/CN/TaskPool.md) 文档
- `DevicePool`: 请参考 [DevicePool](./docs/CN/DevicePool.md) 文档
- `DataBase`: 请参考 [DataBase](./docs/CN/DataBase.md) 文档

### 6.2 Designer
请参考 [Designer](./docs/CN/Designer.md) 文档

### 6.3 Coder
请参考 [Coder](./docs/CN/Coder.md) 文档

### 6.4 Verifier
请参考 [Verifier](./docs/CN/Verifier.md) 文档

### 6.5 Conductor
请参考 [Conductor](./docs/CN/Conductor.md) 文档

### 6.6 SWFT Backend
请参考 [SWFT](./docs/CN/SWFT.md) 文档

### 6.7 Triton Backend
请参考 [Triton](./docs/CN/Triton.md) 文档
