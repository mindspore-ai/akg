---
name: standard-workflow
description: "标准的算子生成工作流，包含设计、编码和验证三个阶段"
level: L1
category: workflow
version: "1.0.0"
license: MIT
structure:
  child_skills:
    - coder-agent
    - verifier-agent
  default_children:
    - coder-agent
    - verifier-agent
---

# 标准算子生成工作流

## 概述

标准工作流是 AKG Agents 中最常用的算子生成流程，适用于大多数常规算子的开发场景。

## 工作流程

### 第1阶段：代码生成
- 使用coder-agent生成初始代码
- 支持多种DSL（CUDA、Triton等）
- 支持多种后端（NVIDIA、AMD等）

### 第2阶段：验证测试
- 使用verifier-agent进行验证
- 检查代码正确性
- 性能profiling

## 适用场景

1. **常规算子**：matmul, conv2d, softmax等
2. **单一后端**：只需要支持一种硬件
3. **标准DSL**：使用常见的编程语言

## 不适用场景

- 复杂的融合算子（建议使用adaptive-evolve）
- 需要多轮迭代优化的算子
- 实验性算子

## 配置示例

```yaml
workflow: standard
agents:
  - coder-agent
  - verifier-agent
metrics:
  - accuracy
  - performance
```

## 成功案例

- ✅ MatMul CUDA实现
- ✅ Softmax Triton实现
- ✅ ReLU多后端实现

## 相关Skill

- **子Skill**: coder-agent, verifier-agent
- **替代方案**: adaptive-evolve (用于复杂场景)

