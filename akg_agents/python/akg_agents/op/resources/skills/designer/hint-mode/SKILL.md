---
name: hint-mode
description: "Sketch 设计中 Hint 模式参数空间配置指南，用于从任务描述中提取参数范围并生成可调优的参数空间配置"
category: guide
version: "1.0.0"
metadata:
  role: designer
---

# Hint 模式：参数空间配置指南

## 概述

Hint 模式用于从任务描述中识别参数范围约束，生成参数空间配置（space_config），支持后续的自动调优。

## Hint 语法格式

### 标准格式
```python
# @hint: param in [val1, val2, ...]         → type='choice', values=[...]
# @hint: param in range(min, max, step=N)   → type='range', min=..., max=..., step=...
# @hint: param = value                      → type='fixed', value=...
# @hint: param in pow2(min_pow, max_pow)    → type='power_of_2', min_pow=..., max_pow=...
# @hint: param in pow of 2                  → type='power_of_2'
```

### 兼容格式
```python
# @range_hint("param", start=min, end=max)  → type='range', min=..., max=...
# @elemwise_hint("param", [val1, val2])     → type='choice', values=[...]
# @elem_hint("param", [val1, val2])         → type='choice', values=[...]
```

### 示例
```python
# @hint: batch_size in pow of 2             → {'type': 'power_of_2', 'min_pow': 3, 'max_pow': 6}  # [8, 16, 32, 64]
# @hint: dim in range(16, 65536)            → {'type': 'range', 'min': 16, 'max': 65536, 'step': 1}
# @hint: BLOCK_M in [64, 128, 256]          → {'type': 'choice', 'values': [64, 128, 256]}
```

## Hint 提取规则

1. **识别所有 hint**：包括被注释掉的 hint（`# @range_hint`），都应识别并提取
2. **完整提取**：如果有多个 hint，必须全部提取，不能遗漏
3. **格式转换**：
   - 装饰器格式（如 `@range_hint("param", st=8, ed=64)`）→ 提取括号内信息
   - 注释格式（如 `# @hint: param in range(16, 65536)`）→ 提取冒号后声明
   - 统一转换为标准的 `SPACE_CONFIG` 字典格式

## 参数空间配置模板

```python
"""参数空间配置"""
import torch  # 或 import mindspore as ms

# ===== 参数空间定义 =====
SPACE_CONFIG = {
    'param1': {'type': 'choice', 'values': [val1, val2, ...]},
    'param2': {'type': 'range', 'min': min_val, 'max': max_val, 'step': step_val},
    'param3': {'type': 'power_of_2', 'min_pow': min_exp, 'max_pow': max_exp},
    # ... 根据 hint 提取的所有参数
}

# ===== 元信息 =====
META_INFO = {
    'op_name': 'op_name',
    'framework': 'torch',  # 或 'mindspore'
    'param_names': ['param1', 'param2', ...]  # 参数名列表，保持顺序！
}

# ===== 输入构造函数 =====
def create_inputs(param1, param2, ...):
    """
    根据参数生成输入
    参数顺序必须与 META_INFO['param_names'] 一致
    """
    ...
    return [tensor1, tensor2, ...]

# ===== 初始化输入函数（可选）=====
def get_init_inputs():
    """如果原始代码有此函数，完整复制"""
    return []  # 或 ["auto"]
```

## BLOCK_SIZE 配置选择原则

### 参数范围较小（如 M in [128, 256]）
- 避免过大 BLOCK_SIZE，推荐 [32, 64, 128]
- 确保最小值至少能容纳一个 BLOCK

### 参数范围较大（如 M in range(128, 8192)）
- 提供多种选择 [64, 128, 256, 512]
- 使用 autotune 适配不同大小

## 边界处理策略

### 选项A：使用 mask（推荐）
- 支持任意 shape
- 在草图注释中说明：`使用 mask，支持任意 shape`

### 选项B：不使用 mask（性能更优）
- 要求 shape 整除 BLOCK_SIZE
- 在草图注释中说明：`M % 64 == 0（不使用 mask）`

## 设计适用范围注释

在草图中必须添加"设计适用范围"注释：

```python
# 设计适用范围：
# - 参数范围：M in [128, 2048], N in [128, 4096]
# - 边界处理：使用 mask / 不使用 mask（需整除）
# - BLOCK_SIZE 配置：[64, 128, 256]
# - 约束条件：如不使用 mask，M % 64 == 0
```

**重要**：
- 无论 sketch 用几维，"设计适用范围"注释中必须对原始输入的每个维度分别说明范围
- 不要只写总元素数的范围，要分别写每个维度的范围
- 如果是 2 的幂次也需要标出
