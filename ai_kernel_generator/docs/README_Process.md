# AI Kernel Generator Process 概念说明

## 概述

AI Kernel Generator 提供了两种核心的处理流程来生成和优化内核代码：**Sequential Process（顺序流程）** 和 **Cycle Process（循环流程）**。这两种流程都遵循相同的三阶段架构，但在错误处理和优化策略上有所不同。

## 三阶段架构

所有流程都基于以下三个核心阶段：

1. **Designer（设计阶段）**：生成算子的设计方案和 AUL 代码
2. **Coder（编码阶段）**：将 AUL 代码转换为具体的内核实现代码
3. **Tester（测试阶段）**：验证生成的内核代码的正确性和性能

## Sequential Process（顺序流程）

### 特点
- **简化设计**：采用线性执行模式，按 Designer → Coder → Tester 的顺序依次执行
- **快速失败**：任何阶段失败时立即退出，不进行重试或回滚
- **轻量级**：无内存机制，适合简单场景和快速验证

### 适用场景
- 快速原型验证
- 简单算子实现
- 开发和调试阶段
- 对稳定性要求不高的场景

### 功能特性
- 支持从指定阶段开始执行（通过 `start_stage` 参数）
- 支持批量处理多个样本
- 目前专门针对 SWFT IR 类型优化
- 提供详细的执行日志和状态报告

### 使用示例
```python
process = SequentialProcess(
    op_name="add_op",
    op_task_str=...,
    backend="Ascend310P3",
    samples_num=2, # 随机采样次数
    record_mode=False,
    ir_type="SWFT"
)

# 从头开始执行完整流程
status = process.run()

# 从编码阶段开始执行
status = process.run(start_stage=ProcessStage.CODER, grouped_aul_codes=aul_codes)
```

## Cycle Process（循环流程）
TODO