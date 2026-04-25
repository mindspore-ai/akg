---
name: pypto-case-elemwise-gelu
description: "模式 A 示例：1D elementwise — GELU 激活，展示展平、无 tanh 时的手写公式、运算符使用"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "elementwise,activation"
---

# 模式 A：1D Elementwise — GELU

```python
def create_gelu_kernel(flat_size):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def gelu_kernel(
        x: pypto.Tensor((flat_size,), pypto.DT_FP32),
    ) -> pypto.Tensor((flat_size,), pypto.DT_FP32):
        output = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        x_cubed = x * x * x
        inner = x + x_cubed * 0.044715
        tanh_arg = inner * 0.7978845608028654
        exp_pos = pypto.exp(tanh_arg * 2.0)
        tanh_val = (exp_pos - 1.0) / (exp_pos + 1.0)
        output[:] = x * 0.5 * (1.0 + tanh_val)
        return output
    return gelu_kernel
```

forward：`reshape(-1)` → kernel → `reshape(x.shape)`

## 模式要点
- forward 中 `assert dim + shape`，`reshape(-1)` 展平为 1D
- `set_vec_tile_shapes(8192)` — 1D 只需一个参数
- 无内建 tanh → `(exp(2x)-1)/(exp(2x)+1)` — 注意 `exp_pos - 1.0` 中 Tensor 在左，合法
- 所有 `Tensor op scalar` 合法；若需 `scalar op Tensor` 则改写
