# KernelBench 任务格式规范

## 文件结构

KernelBench 任务文件是一个 **单一自包含 Python 文件**，包含以下四个必需部分：

### 1. Imports 区

```python
import torch
import torch.nn as nn
# 只允许标准库和 PyTorch 相关包
# 禁止 import 项目内的其他文件
```

### 2. Model 类

```python
class Model(nn.Module):
    def __init__(self, <init_params>):
        super(Model, self).__init__()
        # 保存所有初始化参数

    def forward(self, <forward_inputs>) -> torch.Tensor:
        # 核心计算逻辑
        return output
```

### 3. `get_inputs()` 函数

```python
def get_inputs():
    """返回 forward() 的输入参数列表"""
    return [torch.randn(batch_size, dim)]
```

### 4. `get_init_inputs()` 函数

```python
def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    return [dim_value]
```

## 关键约束

| 约束 | 说明 |
|------|------|
| 自包含 | 所有依赖函数必须内联到文件中 |
| 可执行 | `Model(*get_init_inputs()).forward(*get_inputs())` 必须直接运行 |
| 确定性 | 给定相同输入，输出必须一致 |
| 无 NaN/Inf | forward 输出不能包含 NaN 或 Inf |
| 合理输入 | get_inputs 应提供合理大小的输入（不能过小或过大） |
| 一致返回 | 返回类型/形状必须与原始实现一致 |
