# ModelNew 格式规范 — PyTorch 原生实现

适用于 `dsl` 为 `torch` 的场景（Kernel → PyTorch 转换任务）。

**所有生成的代码必须是 ModelNew 类格式，不使用函数形式。**

## 1. ModelNew 类模板

> **禁止使用自定义 Kernel**（如 triton、cuda_extension）！这是 Kernel → PyTorch 转换任务。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    PyTorch 原生实现（从自定义 Kernel 转换）
    """
    def __init__(self, ...):
        super().__init__()
        # 如果有可学习参数（如 Linear, Conv2d），在此定义
        # 需要设置固定随机种子以确保权重一致
        torch.manual_seed(0)

    def forward(self, ...):
        # 使用 PyTorch 原生操作实现等价功能
        # 常见转换示例：
        # Triton: tl.maximum(x, 0) → torch.relu(x)
        # Triton: tl.sum(x, axis=0) → torch.sum(x, dim=0)
        # CUDA: max(a, b) → torch.maximum(a, b)
        # CUDA: __syncthreads() → (PyTorch 自动同步，无需显式调用)
        return output
```

**注意**：以上 `self` 属性只针对于需要用到内置参数的实现，对于使用 `get_inputs` 或 `get_init_inputs` 传入的参数，需要直接调用。

## 2. 无参数算子

如果算子没有可学习参数（如 ReLU），`__init__` 可以为空：

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
```

## 3. 有参数算子

对于有可学习参数的算子（如 Linear, Conv2d），必须在 `__init__` 中通过固定随机种子构建参数：

```python
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        torch.manual_seed(0)  # 固定种子，确保与原始 Model 权重一致
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.clone())
        self.bias = nn.Parameter(linear.bias.clone()) if linear.bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

**注意**：以上 `self` 属性只针对于需要用到内置参数的实现，对于使用 `get_inputs` 或 `get_init_inputs` 传入的参数，需要直接调用。

## 4. Shape 参数获取

对于任务输入中固定写死的 `init_inputs` 参数，以及 class 外定义的参数，硬编码至代码中。
所需要的 shape 参数，要从 inputs 的数据形状中获取，以适应不同的输入（**重要**：需要仔细检查输入的变量和 shape 与代码中一一对应）：

```python
def forward(self, input_tensor, ...):
    # 硬编码的参数（如果有无法从 inputs 中获取的参数）
    # args = ...  # 无法从 inputs 中获取的参数信息硬编码于此

    # 从输入张量获取 shape 参数
    P1, P2, P3 = input_tensor.shape  # 变量名应该与 inputs 构造时对应的变量保持一致
    ...
```
