# ModelNew 格式规范 — CUDA C + PyTorch

适用于 `dsl` 为 `cuda_c`，`framework` 为 `torch` 的场景。

**所有生成的代码必须是 ModelNew 类格式，不使用函数形式。**

## 1. ModelNew 类模板

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel 代码
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void xxx_kernel(...) {
    // CUDA kernel 实现
    ...
}

torch::Tensor xxx_forward(torch::Tensor input) {
    // 分配输出、启动 kernel
    ...
    xxx_kernel<<<grid, block>>>(...);
    return output;
}
"""

# 加载 CUDA 扩展
xxx_module = load_inline(
    name="xxx_module",
    cpp_sources="",
    cuda_sources=cuda_source,
    functions=["xxx_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        torch.manual_seed(0)
        # 如果有可学习参数，同固定种子 + nn.Parameter 模式

    def forward(self, ...):
        return xxx_module.xxx_forward(...)
```

**注意**：以上 `self` 属性只针对于需要用到内置参数的实现，对于使用 `get_inputs` 或 `get_init_inputs` 传入的参数，需要直接调用。

## 2. 无参数算子

如果算子没有可学习参数（如 ReLU），`__init__` 可以为空：

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xxx_module.xxx_forward(x)
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
        return xxx_module.xxx_forward(x, self.weight, self.bias)
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
    # 执行 kernel 函数
    ...
```
