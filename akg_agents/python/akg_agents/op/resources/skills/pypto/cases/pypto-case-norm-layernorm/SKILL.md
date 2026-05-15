---
name: pypto-case-norm-layernorm
description: "模式 C 示例：2D Norm + Loop — LayerNorm，展示 forward 降维为 2D、kernel 内 sum 归约 + 归一化"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "norm,reduction,loop"
---

# 模式 C-1：2D Norm — LayerNorm

forward 中 `reshape(batch, -1)` 降为 2D，kernel 沿 batch 维 loop。

```python
BASIC_BATCH = 4

def create_layernorm_kernel(batch, hidden, eps):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor((batch, hidden), pypto.DT_FP32),
    ) -> pypto.Tensor((batch, hidden), pypto.DT_FP32):
        output = pypto.tensor([batch, hidden], pypto.DT_FP32)
        num_iters = ceil_div(batch, BASIC_BATCH)
        pypto.set_vec_tile_shapes(1, 16384)
        inv_h = 1.0 / hidden
        for bi in pypto.loop(0, num_iters, 1, name="LOOP_LN", idx_name="bi"):
            offset = bi * BASIC_BATCH
            x_chunk = pypto.view(x, [BASIC_BATCH, hidden], [offset, 0])
            mean = pypto.sum(x_chunk, dim=1, keepdim=True) * inv_h
            var = pypto.sum(x_chunk * x_chunk, dim=1, keepdim=True) * inv_h - mean * mean
            normed = (x_chunk - mean) / pypto.sqrt(var + eps)
            pypto.assemble(normed, [offset, 0], output)
        return output
    return kernel
```

forward：`reshape(B, -1)` → kernel → `reshape(x.shape)`
GroupNorm 同模式：forward 中 `reshape(B*G, -1)` → 2D kernel。

## 模式要点
- `pypto.sum(dim=int)` — dim 只能传单个 int
- mean = `sum * (1/size)` — 没有 mean API
- 方差 = `E[x²] - E[x]²` — 两次 sum 实现
- `set_vec_tile_shapes(1, 16384)` — 2D，第一维小批量，第二维大 tile
