---
name: triton-cuda-error-fix
description: triton-cuda常见错误及修复方法，用于代码生成时避免同类问题
category: fix
version: "1.0.0"
metadata:
  source: error_fix
  case_type: fix
  backend: cuda
  dsl: triton_cuda
---

### 数学函数调用错误

- **报错特征**: `AttributeError: module 'triton.language' has no attribute 'tanh'`
- **修复**:
  - Triton CUDA 后端的数学函数需通过 `tl.extra.cuda.libdevice` 模块调用
  - 避免直接使用 `tl.math.xxx` 或 `tl.xxx`

```python
# 错误：直接使用 tl.tanh
result = 0.5 * x * (1.0 + tl.tanh(inner))

# 正确：通过 libdevice 调用
result = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
```