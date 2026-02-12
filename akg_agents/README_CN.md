[English Version](./README.md)

<div align="center">
  <img src="./akg_agents_logo.jpg" alt="AIKG Logo" width="400">
</div>

<div align="center">

# AKG Agents

</div>

<details>
<summary><b>📋 目录</b></summary>

- [AKG Agents](#akg-agents)
  - [📘 1. 项目简介](#-1-项目简介)
  - [🗓️ 2. 更新日志](#️-2-更新日志)
  - [🛠️ 3. AKG_CLI 快速上手](#️-3-akg_cli-快速上手)
  - [⚙️ 4. 配置](#️-4-配置)
    - [配置快速指南](#配置快速指南)
      - [Step 1: 基础环境配置](#step-1-基础环境配置)
        - [API与模型配置](#api与模型配置)
        - [第三方依赖](#第三方依赖)
      - [Step 2: 后端依赖配置](#step-2-后端依赖配置)
      - [Step 3: 可选工具配置](#step-3-可选工具配置)
        - [Embedding 模型配置（RAG）](#embedding-模型配置rag)
  - [▶️ 5. 教程示例](#️-5-教程示例)
  - [📐 6. 设计文档](#-6-设计文档)
    - [核心框架](#核心框架)
    - [核心组件（Agents & 搜索优化）](#核心组件agents--搜索优化)
    - [知识与数据](#知识与数据)
    - [服务化架构](#服务化架构)
    - [后端支持](#后端支持)

</details>

## 📘 1. 项目简介
AKG Agents 是一个基于大语言模型（LLM）的多 Agent 协作框架。
框架提供通用的 Agent 编排引擎（`core_v2`）：包括 ReAct Agent 基类、Skill 动态知识注入、LangGraph 工作流、树状 Trace 系统等基础能力，可用于构建各类 AI 辅助场景。
当前已落地的首个场景是 **AI 算子生成**（`op`）：通过 LLM 规划和多 Agent 协同，完成多后端、多 DSL 的 AI 算子自动生成与优化。后续将扩展至文档、重构、测试等更多通用场景。

<div align="center" style="background-color:white">
  <img src="./akg_agents.png" alt="AIKG Architecture" width="600">
</div>

## 🗓️ 2. 更新日志
- 2026-02-10：核心框架重构。将通用 Agent 能力与算子场景解耦，构建可复用的多 Agent 协作框架。主要包括：统一配置管理（[`settings.json`](./examples/settings.example.json)）、Agent 基类与注册机制（`AgentBase` / `ReActAgent` / `AgentRegistry`）、[Skill 动态知识注入系统](./docs/CN/SkillSystem.md)、[树状 Trace 系统](./docs/CN/Trace.md)、[LangGraph 工作流](./docs/CN/Workflow.md)、OpenAI 兼容 [Embedding/RAG](./docs/CN/RAG.md)、[Database 基类解耦](./docs/CN/Database.md)。详见《[Refactor 迁移说明](./docs/CN/Refactor.md)》。
- 2025-12-01：引入 LangGraph 重构任务调度系统，新增 `LangGraphTask` 替代原 `Task 任务编排` 方案。详见《[Workflow 文档](./docs/CN/Workflow.md)》。
- 2025-11-25：支持服务化架构，支持`client-server-worker`分离架构，支持各类灵活并发需求，详见《[服务化架构文档](./docs/CN/ServerArchitecture.md)》。
- 2025-10-14：支持 TileLang_CUDA后端代码生成能力。KernelBench Level1 的 TileLang_CUDA后端算子生成成功率结果详见《[基准测试结果](./docs/CN/DSLBenchmarkResults202509.md)》。
- 2025-09-26：支持 CUDA C 与 CPP 后端代码生成能力。KernelBench Level1 的 CUDA C 与 CPP 后端算子生成成功率结果详见《[基准测试结果](./docs/CN/DSLBenchmarkResults202509.md)》。
- 2025-09-14：KernelBench Level1 算子生成成功率更新，详见《[基准测试结果](./docs/CN/BenchmarkResults202509.md)》。
- 2025-08-12：支持"文档驱动式接入"功能（已被 [Skill System](./docs/CN/SkillSystem.md) 替代）。
- 2025-06-27：AIKG 初始版本，支持 Triton 与 SWFT 后端代码生成能力。


## 🛠️ 3. AKG_CLI 快速上手

### 基础安装
```bash
# 1. 环境设置（可选，推荐 Python 3.10/3.11/3.12）
# 使用 conda 环境
conda create -n akg_agents python=3.11
conda activate akg_agents

# 2. 克隆仓库
git clone https://gitcode.com/mindspore/akg.git -b br_akg_agents
cd akg

# 3. 安装依赖
pip install -r akg_agents/requirements.txt
# pip install -r akg_agents/rag_requirements.txt  # 如果需要 RAG 功能（可选）

# 4. 安装 AIKG
pip install -e ./akg_agents --no-build-isolation

# 5. 设置环境变量
cd ./akg_agents
source env.sh
```

### 配置 LLM

推荐使用 `settings.json` 配置（完整示例参见 [`examples/settings.example.json`](./examples/settings.example.json)）：

```bash
# 将示例配置复制到项目配置目录
mkdir -p .akg
cp examples/settings.example.json .akg/settings.json
# 编辑 .akg/settings.json，填入你的 API Key 和模型信息
```

`settings.json` 支持按模型等级（`complex` / `standard` / `fast`）分别配置不同的 LLM：

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

也支持环境变量快速配置（优先级高于 `settings.json`）：

```bash
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"
export AKG_AGENTS_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"  # 或 "disabled"
```

> 💡 配置加载优先级（从高到低）：环境变量 → `.akg/settings.local.json` → `.akg/settings.json` → `~/.akg/settings.json` → 默认值

### 启动 AKG_CLI
```bash
# Ascend 910B2（--framework torch，--dsl triton_ascend/triton_cuda/cuda/tilelang_cuda 等）
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100: --backend cuda --arch a100 --dsl triton_cuda
# akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --devices 0,1,2,3,4,5,6,7
```

### 使用方法
启动 AKG_CLI 后，您可以通过以下方式使用：

1. **直接提问**：例如 "帮我生成个 relu 算子"
2. **提供代码**：粘贴现有的 KernelBench 风格的代码
   - AIKG 会首先编写一个 baseline torch 代码用于结果对比
   - 验证无误后，可以让它生成目标代码（生成的 DSL 类型由启动时的 `--dsl` 参数决定）

> 💡 **提示**：更多使用示例和详细说明，请参考《[AKG_CLI 文档](./docs/CN/AKG_CLI.md)》
>
> 💡 **二次开发**：如果你需要扩展消息类型、接入自定义面板、或理解 MainOpAgent/子 Agent 与 TUI 的交互机制，请参考《[AKG_CLI 二次开发指南](./docs/CN/AKG_CLI_Develop.md)》



## ⚙️ 4. 配置

### 配置快速指南

#### Step 1: 基础环境配置

##### API与模型配置

AIKG 采用多层级配置系统管理 LLM 服务。推荐使用 `settings.json`，也兼容环境变量。

**方式一：`settings.json`（推荐）**

将 [`examples/settings.example.json`](./examples/settings.example.json) 复制到 `.akg/settings.json` 并编辑：

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

**方式二：环境变量（快速启动 / CI 场景）**

```bash
# 单模型配置（自动应用到所有模型等级：complex/standard/fast）
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"

# 按等级分别配置（可选）
export AKG_AGENTS_COMPLEX_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_COMPLEX_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_COMPLEX_MODEL_NAME="gpt-4"
```

> 💡 配置加载优先级：环境变量 > `.akg/settings.local.json` > `.akg/settings.json` > `~/.akg/settings.json` > 默认值

更多配置选项：
- **工作流配置**：基于 LangGraph 的 Python 定义工作流，支持图结构可视化与类型安全状态管理。详见《[Workflow 文档](./docs/CN/Workflow.md)》。
- **Skill System**：通过技能注册和智能选择，为 Agent 动态注入领域知识（替代旧的 `docs_dir` 文档驱动方案）。详见《[Skill System 文档](./docs/CN/SkillSystem.md)》。

详细配置说明请参考 [API配置文档](./docs/CN/API.md)。

##### 第三方依赖
本项目使用 git submodule 管理部分第三方依赖（如： Kernelbench、MultiKernelbench等）。

初次克隆或拉取更新后，请使用以下命令初始化并下载 `akg_agents` 相关的依赖：
```bash
# 初始化并拉取 akg_agents 相关的子模块
git submodule update --init "akg_agents/thirdparty/*"
```

#### Step 2: 后端依赖配置
根据您的硬件平台选择相应的后端：

| 平台 | 后端 | 参考链接 |
|------|------|----------|
| 华为Atlas A2训练系列产品 | Triton | https://gitee.com/ascend/triton-ascend |
| NVIDIA GPU | Triton | https://github.com/triton-lang/triton |
| NVIDIA GPU | TileLang | https://github.com/tile-ai/tilelang |
| 华为Atlas A2训练系列产品 | TileLang | https://github.com/tile-ai/tilelang |
| NVIDIA GPU | CUDA C/C++ | https://docs.nvidia.com/cuda/ |

#### Step 3: 可选工具配置

##### Embedding 模型配置（RAG）

RAG 检索功能需要 Embedding 模型生成向量表示。系统支持 **远程 API**（推荐）和 **本地模型** 两种方式。

**方式一：远程 Embedding API（推荐）**

在 `settings.json` 中配置 `embedding` 字段，或设置环境变量：

```bash
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export AKG_AGENTS_EMBEDDING_API_KEY="YOUR_API_KEY"
```

**方式二：本地 HuggingFace 模型**

设置本地模型路径：

```bash
export EMBEDDING_MODEL_PATH="/path/to/your/embedding-model"
```

> 💡 加载优先级：远程 API 配置 → 本地 `EMBEDDING_MODEL_PATH` → 禁用向量检索功能

详见《[RAG 使用指南](./docs/CN/RAG_Usage.md)》和《[RAG 文档](./docs/CN/RAG.md)》。

> 💡 **配置提示**: 
> - 详细的API配置请参考 [API文档](./docs/CN/API.md) 
> - 数据库配置请参考 [DataBase文档](./docs/CN/Database.md)
> - 更多配置选项请参考各组件的专门文档


## ▶️ 5. 教程示例

以下为 `examples/` 目录中的常用示例：

| 示例 | 说明 |
|------|------|
| `run_torch_npu_triton_single.py` | 单算子示例（Torch + Triton，Ascend）。 |
| `run_torch_cpu_cpp_single.py` | CPU C++ 单算子示例（Torch + CPP）。 |
| `run_cudac_to_torch_single.py` | CUDA C 单算子示例（Torch + CUDA C）。 |
| `run_triton_to_torch_single.py` | Triton CUDA 单算子示例（Torch + Triton CUDA）。 |
| `run_kernel_agent.py` | KernelAgent（ReAct Agent）交互式调用示例。 |
| `run_torch_evolve_triton.py` | 进化算法算子优化示例（Torch + Triton）。 |
| `run_cuda_to_ascend_conversion.py` | CUDA 到 Ascend 算子转换示例。 |
| `run_client_server_worker.py` | Client-Server 分布式运行示例。 |
| `kernel_profile.py` | 算子性能 Profiling 示例。 |
| `handwrite_optimization_analyzer.py` | 手写优化分析器示例。 |

更多上手流程与参数说明，请参考《[Tutorial](./docs/CN/Tutorial.md)》。


## 📐 6. 设计文档

> 建议先阅读《[Workflow 文档](./docs/CN/Workflow.md)》了解 LangGraph 工作流与任务编排方案，再阅读《[Skill System](./docs/CN/SkillSystem.md)》了解知识管理与动态注入机制。

### 核心框架
- **[Workflow 与任务系统](./docs/CN/Workflow.md)** - 基于 LangGraph 的工作流与任务管理（`LangGraphTask`）
- **[Trace System](./docs/CN/Trace.md)** - 树状推理追踪系统（支持多分叉、断点续跑）
- **[Skill System](./docs/CN/SkillSystem.md)** - 技能管理与动态知识注入系统
- **[TaskPool](./docs/CN/TaskPool.md)** - 任务池管理
- **[DevicePool](./docs/CN/DevicePool.md)** - 设备池管理

### 核心组件（Agents & 搜索优化）
- **[KernelDesigner](./docs/CN/KernelDesigner.md)** - 算法草图设计 Agent（`call_kernel_designer`）
- **[KernelGen](./docs/CN/KernelGen.md)** - 内核代码生成 Agent（`call_kernel_gen`）
- **[Evolve 进化优化](./docs/CN/Evolve.md)** - 基于遗传算法的算子进化生成与优化
- **[Adaptive Search 自适应搜索](./docs/CN/Search.md)** - 基于 UCB 策略的异步流水线搜索框架
- **[Refactor 迁移说明](./docs/CN/Refactor.md)** - 模块迁移重构文档

### 知识与数据
- **[Database](./docs/CN/Database.md)** - 数据库模块（基类 + 算子专用 `CoderDatabase`）
- **[RAG](./docs/CN/RAG.md)** - 向量检索增强生成模块
- **[RAG 使用指南](./docs/CN/RAG_Usage.md)** - RAG 配置与使用教程
- **[Skill 贡献指南](./docs/CN/SkillContributionGuide.md)** - 如何贡献新的 Skill

### 服务化架构
- **[Server Architecture](./docs/CN/ServerArchitecture.md)** - 服务化架构文档，包含 Client-Server-Worker 架构、WorkerManager 负载均衡、便捷函数使用等

### 后端支持
- **[Triton Backend (Ascend/CUDA)](./docs/CN/Triton.md)** - Triton 计算后端
- **[TileLang Backend (CUDA)](./docs/CN/DSLBenchmarkResults202509.md)** - TileLang 计算后端
- **[CUDA C/C++ Backend](./docs/CN/DSLBenchmarkResults202509.md)** - CUDA Native 后端
- **[CPU Backend](./docs/CN/DSLBenchmarkResults202509.md)** - CPU 后端
