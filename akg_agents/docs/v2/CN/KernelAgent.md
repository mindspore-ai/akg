[English Version](../KernelAgent.md)

# AKG Kernel Agent

## 1. 概述

AKG Kernel Agent 是 AKG Agents 的首个落地场景，专注于**多后端、多 DSL 的高性能算子代码生成与优化**。

- **CLI 入口**：`akg_cli op`
- **领域**：面向 AI 加速器的高性能算子（kernel）代码生成

## 2. 支持的后端与 DSL

| 平台 | 后端 | DSL | 示例 |
|------|------|-----|------|
| 华为 Atlas A2 训练系列产品 | Ascend | Triton Ascend | `--backend ascend --dsl triton_ascend` |
| NVIDIA GPU | CUDA | Triton CUDA | `--backend cuda --dsl triton_cuda` |
| NVIDIA GPU | CUDA | CUDA C | `--backend cuda --dsl cuda` |
| NVIDIA GPU | CUDA | TileLang CUDA | `--backend cuda --dsl tilelang_cuda` |
| CPU | CPU | C++ | `--backend cpu --dsl cpp` |

## 3. 内置工作流

Kernel Agent 支持多种工作流策略：

| 工作流 | 说明 |
|--------|------|
| **Default** | 完整流水线：Designer → Coder ↔ Verifier |
| **CoderOnly** | 仅代码生成（跳过设计阶段） |
| **Evolve** | 基于进化算法的算子优化 |
| **AdaptiveSearch** | 基于 UCB 的异步流水线搜索 |
| **KernelGenOnly** | 仅生成算子，不验证 |
| **VerifierOnly** | 仅验证（用于已有代码） |

## 4. 核心 Agents

### KernelDesigner

算法草图设计 Agent。分析算子需求，生成高层次算法设计和优化提示。

- 基于 Skill：动态注入相关领域知识
- 支持 DSL 特定的设计模式

### KernelGen

算子代码生成 Agent。基于算法设计生成目标 DSL 的可执行算子代码。

- 基于 Skill：使用 DSL 特定的编码技能
- 可作为工具被其他 Agent 调用

### TaskConstructor

标准化任务构建 Agent。从用户输入（如 PyTorch 代码）中提取并标准化算子定义为结构化任务格式。

## 5. 工作流：Default 流水线

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Designer   │────▶│    Coder     │────▶│   Verifier   │
│              │     │  (KernelGen) │     │              │
│ 算法设计      │     │ 代码生成      │     │ 正确性检查    │
│              │     │              │◀────│              │
└──────────────┘     └──────────────┘     └──────────────┘
                           │                     │
                           │                     ▼
                           │              ┌──────────────┐
                           │              │   Profiler   │
                           │              │ 性能分析      │
                           │              │              │
                           └──────────────┴──────────────┘
```

1. **Designer** 分析算子需求，生成算法草图
2. **Coder** 基于设计生成算子代码
3. **Verifier** 通过对比框架实现检查正确性
4. 如果验证失败，Coder 收到错误反馈并重试
5. **Profiler** 测量性能（执行时间、加速比）

## 6. DSL 配置

每个 DSL 后端有一个 YAML 配置文件控制工作流行为：

```yaml
# 关键配置字段
agent_model_config:
  kernel_designer: "complex"
  kernel_gen: "standard"

log_dir: "logs/"
default_workflow: "default"

profile_settings:
  run_times: 50
  warmup_times: 5

verify_timeout: 300
```

## 7. 快速开始

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 \
  --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA A100
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0,1,2,3,4,5,6,7
```

启动后，你可以：
1. 描述需求："帮我生成一个 relu 算子"
2. 粘贴 KernelBench 风格的 PyTorch 代码进行转换

更多 CLI 详情请参考 [AKG CLI](./AKG_CLI.md)。
