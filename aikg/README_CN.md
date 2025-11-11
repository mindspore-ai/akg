[English Version](./README.md)

<div align="center">
  <img src="./aikg_logo.jpg" alt="AIKG Logo" width="400">
</div>

<div align="center">

# AI-driven Kernel Generator(AIKG)

</div>

<details>
<summary><b>📋 目录</b></summary>

- [AI-driven Kernel Generator(AIKG)](#ai-driven-kernel-generatoraikg)
  - [📘 1. 项目简介](#-1-项目简介)
  - [🗓️ 2. 更新日志](#️-2-更新日志)
  - [🛠️ 3. 安装部署流程](#️-3-安装部署流程)
  - [⚙️ 4. 配置](#️-4-配置)
    - [配置快速指南](#配置快速指南)
      - [Step 1: 基础环境配置](#step-1-基础环境配置)
        - [API与模型配置](#api与模型配置)
        - [第三方依赖](#第三方依赖)
      - [Step 2: 前端依赖配置](#step-2-前端依赖配置)
        - [MindSpore 2.7版本 前端依赖（可选）](#mindspore-27版本-前端依赖可选)
      - [Step 3: 后端依赖配置](#step-3-后端依赖配置)
      - [Step 4: 可选工具配置](#step-4-可选工具配置)
        - [文本相似性检测依赖](#文本相似性检测依赖)
  - [▶️ 5. 教程示例](#️-5-教程示例)
  - [📐 6. 设计文档](#-6-设计文档)
    - [核心框架](#核心框架)
    - [核心组件](#核心组件)
    - [后端支持](#后端支持)

</details>

## 📘 1. 项目简介
AIKG 是一款 AI 驱动的算子生成器。
AIKG 利用大语言模型(LLM)的代码生成能力，通过大语言模型规划和控制（多个）Agent 协同完成多后端、多类型的AI算子生成和自动优化。
同时 AIKG 提供丰富的算子Agent相关子模块，用户可组合构建自定义算子Agents任务。

## 🗓️ 2. 更新日志
- 2025-10-14：支持 TileLang_CUDA后端代码生成能力。KernelBench Level1 的 TileLang_CUDA后端算子生成成功率结果详见《[基准测试结果](./docs/CN/DSLBenchmarkResults202509.md)》。
- 2025-09-26：支持 CUDA C 与 CPP 后端代码生成能力。KernelBench Level1 的 CUDA C 与 CPP 后端算子生成成功率结果详见《[基准测试结果](./docs/CN/DSLBenchmarkResults202509.md)》。
- 2025-09-14：KernelBench Level1 算子生成成功率更新，详见《[基准测试结果](./docs/CN/BenchmarkResults202509.md)》。
- 2025-08-12：支持"文档驱动式接入"功能，按统一文档规范提供资料即可快速、灵活地接入新的 DSL/前端/后端（详见《[文档驱动式接入指南](./docs/CN/DocDrivenIntegration.md)》）。
- 2025-06-27：AIKG 初始版本，支持 Triton 与 SWFT 后端代码生成能力。


## 🛠️ 3. 安装部署流程
```bash
# 1. 环境设置
# 1.1 使用conda环境（可选， 推荐python3.9/3.10/3.11版本）
conda create -n aikg python=3.11
conda activate aikg

# 1.2 或者创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/active

# 2. pip安装依赖
pip install -r requirements.txt

# 3. whl安装/环境设置
# 3.1 whl安装
bash build.sh
pip install output/ai_kernel_generator-*-py3-none-any.whl

# 3.2 或者设置环境变量
cd aikg
source env.sh
```


## ⚙️ 4. 配置

### 配置快速指南

#### Step 1: 基础环境配置

##### API与模型配置
AIKG 通过环境变量来设置不同大语言模型（LLM）服务的 API。请根据您使用的服务，配置相应的环境变量：

```bash
# 各厂商API接口。详细支持列表请参考docs/API.md
export AIKG_XXX_API_KEY=xxx

# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434
```
更多配置选项：
- **任务编排方案配置（Task Orchestration Plan Configuration）**: 声明一次任务的完整运行方案（包含 `agent_model_config`、`workflow_config_path`、`docs_dir` 等）。常见方案文件：`default_triton_cuda_config.yaml`、`default_triton_ascend_config.yaml`、`vllm_triton_coderonly_config.yaml`。详见《[任务编排方案配置](./docs/CN/TaskOrchestrationPlan.md)》。
- **模型配置**: `llm_config.yaml` 中预设了多种 LLM 服务商的模型配置（DeepSeek、Qwen、Moonshot 等）。编排配置中的 `agent_model_config` 取值来源于该文件的预设名称。
- **工作流定义（Workflow）**: 通过 `workflow_config_path` 指定工作流 YAML，定义 Agent 执行顺序与约束，支持 `default_workflow.yaml`、`coder_only_workflow.yaml` 等。详见《[工作流系统设计文档](./docs/CN/Workflow.md)》。
- **文档驱动式接入（Doc-Driven Integration）**: 通过编排配置的 `docs_dir` 为各 Agent 提供参考文档目录。详见《[文档驱动式接入指南](./docs/CN/DocDrivenIntegration.md)》。

详细配置说明请参考 [API配置文档](./docs/CN/API.md)。

##### 第三方依赖
本项目使用 git submodule 管理部分第三方依赖。

初次克隆或拉取更新后，请使用以下命令初始化并下载 `aikg` 相关的依赖：
```bash
# 初始化并拉取 aikg 相关的子模块
git submodule update --init "aikg/thirdparty/*"
```

#### Step 2: 前端依赖配置

##### MindSpore 2.7版本 前端依赖（可选）
支持python版本：python3.11、python3.10、python3.9
支持系统版本：aarch64、x86_64
推荐按官方安装指南选择环境与安装方式：[MindSpore 2.7 安装指南](https://www.mindspore.cn/install)
```bash
pip install mindspore==2.7.0 -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
```

#### Step 3: 后端依赖配置
根据您的硬件平台选择相应的后端：

| 平台 | 后端 | 参考链接 |
|------|------|----------|
| 华为Atlas A2训练系列产品 | Triton | https://gitee.com/ascend/triton-ascend |
| NVIDIA GPU | Triton | https://github.com/triton-lang/triton |
| 华为Atlas推理系列产品 | SWFT | https://gitee.com/mindspore/akg/tree/br_aikg/swft |

#### Step 4: 可选工具配置

##### 文本相似性检测依赖
文本句子相似性检测工具text2vec-large-chinese： 若无法自动加载模型，需要手动下载到thirdparty目录下
将下载后的模型地址添加到database对应的yaml中，请参考  [DataBase](./docs/CN/DataBase.md) 文档
```bash
bash download.sh --with_local_model
```

> 💡 **配置提示**: 
> - 详细的API配置请参考 [API文档](./docs/CN/API.md) 
> - 数据库配置请参考 [DataBase文档](./docs/CN/DataBase.md)
> - 更多配置选项请参考各组件的专门文档


## ▶️ 5. 教程示例

以下为 `examples/` 目录中的常用示例：

| 示例 | 说明 |
|------|------|
| `run_mindspore_triton_single.py` | 单算子示例（MindSpore + Triton，Ascend 910B4）。 |
| `run_mindspore_triton_parallel.py` | 并行多算子示例（MindSpore + Triton，Ascend 910B4）。 |
| `run_numpy_swft_relu.py` | SWFT ReLU 示例（Ascend 310P3）。 |
| `run_numpy_swft_swiglu.py` | SWFT SwiGLU 示例（Ascend 310P3）。 |

更多上手流程与参数说明，请参考《[Tutorial](./docs/CN/Tutorial.md)》。


## 📐 6. 设计文档

> 建议先阅读《[任务编排方案配置](./docs/CN/TaskOrchestrationPlan.md)》，了解任务运行方案与入口；工作流细节见《[Workflow](./docs/CN/Workflow.md)》，文档规范见《[文档驱动式接入指南](./docs/CN/DocDrivenIntegration.md)》。

### 核心框架
- **[Task](./docs/CN/Task.md)** - 任务管理模块
- **[Trace](./docs/CN/Trace.md)** - 执行追踪模块  
- **[TaskPool](./docs/CN/TaskPool.md)** - 任务池管理
- **[DevicePool](./docs/CN/DevicePool.md)** - 设备池管理
- **[DataBase](./docs/CN/DataBase.md)** - 数据库模块

### 核心组件
- **[Designer](./docs/CN/Designer.md)** - 算子设计器
- **[Coder](./docs/CN/Coder.md)** - 代码生成器
- **[Verifier](./docs/CN/Verifier.md)** - 验证器
- **[Conductor](./docs/CN/Conductor.md)** - 任务编排器

### 后端支持
- **[SWFT Backend](./docs/CN/SWFT.md)** - 华为Atlas推理系列后端
- **[Triton Backend](./docs/CN/Triton.md)** - Triton计算后端
