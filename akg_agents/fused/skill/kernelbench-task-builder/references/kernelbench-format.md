# KernelBench 任务格式规范

## 文件结构

一个合格的 KernelBench 任务文件必须包含以下四个部分：

### 1. Import 和辅助函数

```python
import torch
import torch.nn as nn

# 所有被 Model 依赖的函数必须内联在此处
# 不能 import 外部文件或非标准库模块
```

### 2. Model 类

```python
class Model(nn.Module):
    def __init__(self, param1, param2, ...):
        super(Model, self).__init__()
        # 保存参数

    def forward(self, input1, input2, ...) -> torch.Tensor:
        # 核心计算逻辑
        return output
```

### 3. 输入生成函数

```python
def get_inputs():
    """返回 forward() 的参数列表"""
    return [torch.randn(batch, dim), torch.randn(batch, dim)]
```

### 4. 初始化参数函数

```python
def get_init_inputs():
    """返回 __init__() 的参数列表"""
    return [hidden_dim, num_heads]
```

## 关键约束

1. **自包含**: 所有函数必须内联，不能依赖外部文件
2. **可执行**: `Model(*get_init_inputs()).forward(*get_inputs())` 必须能直接运行
3. **确定性**: 多次运行结果应一致（避免随机种子问题）
4. **无 NaN/Inf**: 输出不能包含 NaN 或 Inf 值
5. **合理输入**: `get_inputs()` 应使用实际场景的输入形状
