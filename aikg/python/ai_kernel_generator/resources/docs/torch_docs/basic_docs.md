# Triton → PyTorch 转换指南

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

## 核心转换对照

| Triton | PyTorch |
|--------|---------|
| `tl.maximum(a, b)` | `torch.maximum(a, b)` |
| `tl.exp(x)` | `torch.exp(x)` |
| `tl.sum(x, axis=0)` | `torch.sum(x, dim=0)` |
| `tl.max(x, axis=0)` | `torch.max(x, dim=0).values` |
| `tl.dot(a, b)` | `torch.matmul(a, b)` |
| `tl.where(c, a, b)` | `torch.where(c, a, b)` |

## 注意事项

1. **禁止使用 Triton** - 不能 `import triton`
2. **保持数值一致** - 输出结果需与原实现一致
3. **设备兼容** - 输入输出设备保持一致
