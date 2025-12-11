# Kernel → PyTorch 转换指南

## 输出代码格式

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args):
        # 使用 PyTorch 原生操作
        return output
```

## Triton → PyTorch 转换对照

| Triton | PyTorch |
|--------|---------|
| `tl.maximum(a, b)` | `torch.maximum(a, b)` |
| `tl.exp(x)` | `torch.exp(x)` |
| `tl.sum(x, axis=0)` | `torch.sum(x, dim=0)` |
| `tl.max(x, axis=0)` | `torch.max(x, dim=0).values` |
| `tl.dot(a, b)` | `torch.matmul(a, b)` |
| `tl.where(c, a, b)` | `torch.where(c, a, b)` |

## CUDA C → PyTorch 转换对照

| CUDA C | PyTorch |
|--------|---------|
| `max(a, b)` / `fmaxf(a, b)` | `torch.maximum(a, b)` |
| `expf(x)` | `torch.exp(x)` |
| `__syncthreads()` | 无需显式调用（PyTorch 自动同步） |
| `atomicAdd()` | `tensor.scatter_add_()` 或原生归约 |
| `blockIdx.x * blockDim.x + threadIdx.x` | 向量化操作（无需手动索引） |
| `load_inline(...)` | 删除，使用原生 PyTorch |

## 注意事项

1. **禁止使用自定义 Kernel** - 不能 `import triton` 或 `torch.utils.cpp_extension.load_inline`
2. **保持数值一致** - 输出结果需与原实现一致
3. **设备兼容** - 输入输出设备保持一致
