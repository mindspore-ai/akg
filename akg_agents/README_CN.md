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
  - [🛠️ 3. 快速上手](#️-3-快速上手)
    - [安装](#安装)
    - [配置 LLM](#配置-llm)
    - [启动 AKG\_CLI](#启动-akg_cli)
    - [使用方法](#使用方法)
  - [▶️ 4. 教程示例](#️-4-教程示例)
  - [📐 5. 设计文档](#-5-设计文档)

</details>

## 📘 1. 项目简介
**AKG Agents** 是一个面向 AI Infra 与高性能计算场景的 LLM 多 Agent 协作框架，致力于通过智能 Agent 协同提升高性能代码的开发与优化效率。

框架提供完整的 Agent 基础设施：包括 ReAct Agent 基类、可扩展的 **Skill / Tools / SubAgent** 机制、LangGraph 工作流编排、树状 Trace 追踪系统，以及统一的配置与注册体系。开发者可以基于这些能力快速构建、组合和部署面向不同任务的智能 Agent。

当前已落地场景为 **AI 算子代码生成**：通过 LLM 规划与多 Agent 协同，实现多后端、多 DSL 的高性能算子自动生成与优化。后续将持续拓展至算子迁移、性能调优、代码重构等更多 AI Infra 相关场景。

## 🗓️ 2. 更新日志
- 2026-02-10：核心框架重构（v2）。将通用 Agent 能力与算子场景解耦，构建可复用的多 Agent 协作框架。详见 [框架架构](./docs/v2/CN/Architecture.md)、[Agent 体系](./docs/v2/CN/AgentSystem.md)、[Skill 系统](./docs/v2/CN/SkillSystem.md)、[工作流](./docs/v2/CN/Workflow.md)、[Trace 系统](./docs/v2/CN/Trace.md)、[配置系统](./docs/v2/CN/Configuration.md)。
- 2025-12-01：引入 LangGraph 重构任务调度系统，新增 `LangGraphTask` 替代原 `Task 任务编排` 方案。详见《[Workflow 文档](./docs/v2/CN/Workflow.md)》。
- 2025-11-25：支持服务化架构，支持`client-server-worker`分离架构，详见《[服务化架构文档](./docs/v1/CN/ServerArchitecture.md)》。
- 2025-10-14：支持 TileLang_CUDA 后端代码生成能力。详见《[基准测试结果](./docs/v1/CN/DSLBenchmarkResults202509.md)》。
- 2025-09-26：支持 CUDA C 与 CPP 后端代码生成能力。详见《[基准测试结果](./docs/v1/CN/DSLBenchmarkResults202509.md)》。
- 2025-09-14：KernelBench Level1 算子生成成功率更新，详见《[基准测试结果](./docs/v1/CN/BenchmarkResults202509.md)》。
- 2025-08-12：支持"文档驱动式接入"功能（已被 [Skill System](./docs/v2/CN/SkillSystem.md) 替代）。
- 2025-06-27：AIKG 初始版本，支持 Triton 与 SWFT 后端代码生成能力。


## 🛠️ 3. 快速上手

### 安装
```bash
# 1. 环境设置（可选，推荐 Python 3.10/3.11/3.12）
conda create -n akg_agents python=3.11
conda activate akg_agents

# 2. 克隆仓库
git clone https://gitcode.com/mindspore/akg.git -b br_agents
cd akg

# 3. 安装依赖
pip install -r akg_agents/requirements.txt

# 4. 安装 AIKG
pip install -e ./akg_agents --no-build-isolation

# 5. 初始化第三方子模块（KernelBench 等，按需）
git submodule update --init "akg_agents/thirdparty/*"
```

### 配置 LLM

将示例配置复制到 `~/.akg/settings.json`，填入你的 API Key 和模型信息：

```bash
mkdir -p ~/.akg
cp akg_agents/examples/settings.example.json ~/.akg/settings.json
```

最简配置只需填写一个模型（自动应用到所有等级）：

```json
{
  "models": {
    "standard": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "YOUR_API_KEY",
      "model_name": "deepseek-chat"
    }
  },
  "default_model": "standard"
}
```

> 如需按等级（`complex` / `standard` / `fast`）分别配置不同模型、配置不同 provider 的 thinking/reasoning 参数、配置 Embedding/RAG、或使用环境变量，请参考 [配置系统文档](./docs/v2/CN/Configuration.md)、基础示例 [`settings.example.json`](./examples/settings.example.json) 和多 provider 示例 [`settings.example.more.json`](./examples/settings.example.more.json)。

### 后端依赖

当前 `br_agents` 分支支持以下三种 DSL，其他后端待适配：

| 平台 | 后端 (DSL) | 参考链接 |
|------|------------|----------|
| 华为 Atlas A2 训练系列产品 | Triton | https://gitee.com/ascend/triton-ascend |
| NVIDIA GPU | Triton | https://github.com/triton-lang/triton |
| CPU (x86_64) | C++ | GCC / Clang |

### 启动与使用

以算子代码生成任务（`akg_cli op`）为例：

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 \
  --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100
# akg_cli op --framework torch --backend cuda --arch a100 \
#   --dsl triton_cuda --devices 0,1,2,3,4,5,6,7
```

启动后，可以通过以下方式交互：

1. **直接描述需求**：例如 "帮我生成个 relu 算子"
2. **提供代码**：粘贴 KernelBench 风格的 PyTorch 代码，AIKG 会自动生成对应 DSL 的算子实现并验证正确性

> `akg_cli` 还支持其他任务类型，完整使用说明请参考《[AKG CLI 文档](./docs/v2/CN/AKG_CLI.md)》。


## ▶️ 4. 教程示例

<details open>
<summary><b>examples/ 目录</b></summary>

| 示例 | 类别 | 说明 |
|------|------|------|
| **NPU** | | |
| `kernel_related/run_torch_npu_triton_single.py` | Kernel | 单算子生成（Torch + Triton Ascend） |
| `kernel_related/run_torch_adaptive_search_triton_ascend.py` | Kernel | UCB 自适应搜索（Torch + Triton Ascend） |
| `kernel_related/run_torch_evolve_triton_ascend.py` | Kernel | 进化算法算子优化（Torch + Triton Ascend） |
| `kernel_related/run_cuda_to_ascend_conversion.py` | Kernel | CUDA 到 Ascend 算子转换 |
| `kernel_related/run_cuda_to_ascend_evolve.py` | Kernel | CUDA 到 Ascend 进化优化 |
| **GPU** | | |
| `kernel_related/gpu/run_triton_to_torch_single.py` | Kernel | 单算子生成（Torch + Triton CUDA） |
| `kernel_related/gpu/run_torch_evolve_triton.py` | Kernel | 进化算法算子优化（Torch + Triton CUDA） |
| `kernel_related/gpu/run_cudac_to_torch_single.py` | Kernel | 单算子生成（Torch + CUDA C） |
| **CPU** | | |
| `kernel_related/cpu/run_torch_cpu_cpp_single.py` | Kernel | 单算子生成（Torch + CPP） |
| `kernel_related/cpu/run_torch_evolve_cpu_cpp.py` | Kernel | 进化算法算子优化（Torch + CPP） |
| `kernel_related/cpu/run_torch_adaptive_search_cpu_cpp.py` | Kernel | UCB 自适应搜索（Torch + CPP） |
| **通用工具** | | |
| `kernel_related/run_kernel_agent.py` | Kernel | KernelAgent（ReAct Agent）交互式调用 |
| `kernel_related/run_kernel_profile.py` | Kernel | 算子性能 Profiling |
| `run_skill/` | Skill | Skill 加载、注册、层级、版本、安装、LLM 选择等示例 |
| `build_a_simple_react_agent/` | 框架 | 基于框架构建自定义 ReAct Agent |
| `build_a_simple_workflow/` | 框架 | 基于 LangGraph 构建自定义 Workflow |
| `settings.example.json` | 配置 | `settings.json` 基础配置模板 |
| `settings.example.more.json` | 配置 | 多 provider 配置示例（OpenAI、DeepSeek、Claude、通义千问、Kimi、豆包等） |

</details>


## 📐 5. 设计文档

> 建议先阅读《[框架架构](./docs/v2/CN/Architecture.md)》了解整体架构，再阅读《[Workflow 文档](./docs/v2/CN/Workflow.md)》和《[Skill 系统](./docs/v2/CN/SkillSystem.md)》了解核心机制。

### 核心框架
- **[框架架构](./docs/v2/CN/Architecture.md)** - 整体架构、模块概览
- **[Agent 体系](./docs/v2/CN/AgentSystem.md)** - Agent 基类、ReAct Agent、注册机制
- **[Skill 系统](./docs/v2/CN/SkillSystem.md)** - 技能管理与动态知识注入
- **[Tools 体系](./docs/v2/CN/Tools.md)** - 工具执行框架、内置工具、领域工具
- **[工作流](./docs/v2/CN/Workflow.md)** - 基于 LangGraph 的工作流编排
- **[Trace 系统](./docs/v2/CN/Trace.md)** - 树状推理追踪系统（多分叉、断点续跑）
- **[配置系统](./docs/v2/CN/Configuration.md)** - 统一配置管理（settings.json / 环境变量）
- **[LLM 接入](./docs/v2/CN/LLM.md)** - LLM 提供者、客户端、Embedding

### 场景
- **[Kernel Agent](./docs/v2/CN/KernelAgent.md)** - 多后端多 DSL 算子代码生成与优化（`akg_cli op`）

### CLI
- **[AKG CLI](./docs/v2/CN/AKG_CLI.md)** - 命令行工具使用指南

### 贡献
- **[Skill 贡献指南](./docs/v2/CN/SkillContributionGuide.md)** - 如何贡献新的 Skill

### 其他模块（v1 文档）
- **[Database](./docs/v1/CN/Database.md)** - 数据库模块
- **[RAG](./docs/v1/CN/RAG.md)** - 向量检索增强生成
- **[RAG 使用指南](./docs/v1/CN/RAG_Usage.md)** - RAG 配置与使用教程
- **[Server Architecture](./docs/v1/CN/ServerArchitecture.md)** - 服务化架构（Client-Server-Worker）
- **[TaskPool](./docs/v1/CN/TaskPool.md)** - 任务池管理
- **[DevicePool](./docs/v1/CN/DevicePool.md)** - 设备池管理
