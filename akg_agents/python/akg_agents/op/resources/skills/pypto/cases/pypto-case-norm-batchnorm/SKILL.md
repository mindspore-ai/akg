---
name: pypto-case-norm-batchnorm
description: "模式 C 示例：3D Norm — BatchNorm，展示 3D 降维、连续单轴 sum 多维归约、expand_clone 广播"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "norm,reduction,loop,expand_clone"
---

# 模式 C-2：3D Norm — BatchNorm

forward 中 `reshape(B, C, -1)` 降为 3D，kernel 沿 channel 维 loop。

```python
BASIC_CHANNEL = 8
MAIN_CHANNEL_LOOP = 8   # channels / BASIC_CHANNEL

def create_batchnorm_kernel(batch, channels, spatial, eps):
    assert channels == MAIN_CHANNEL_LOOP * BASIC_CHANNEL
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor((batch, channels, spatial), pypto.DT_FP32),
    ) -> pypto.Tensor((batch, channels, spatial), pypto.DT_FP32):
        output = pypto.tensor([batch, channels, spatial], pypto.DT_FP32)
        inv_total = 1.0 / (batch * spatial)
        pypto.set_vec_tile_shapes(1, 1, 16384)
        for ci in pypto.loop(0, MAIN_CHANNEL_LOOP, 1, name="LOOP_CH", idx_name="ci"):
            ch_off = ci * BASIC_CHANNEL
            x_chunk = pypto.view(x, [batch, BASIC_CHANNEL, spatial], [0, ch_off, 0])
            # 多轴归约：连续两次单轴 sum
            s = pypto.sum(x_chunk, dim=2, keepdim=True)
            s = pypto.sum(s, dim=0, keepdim=True)      # (1, C, 1)
            sq = pypto.sum(x_chunk * x_chunk, dim=2, keepdim=True)
            sq = pypto.sum(sq, dim=0, keepdim=True)
            mean = s * inv_total
            var = sq * inv_total - mean * mean
            denom = pypto.sqrt(var + eps)
            # expand_clone 广播回 batch 维
            mean_b = pypto.expand_clone(mean, [batch, BASIC_CHANNEL, 1])
            denom_b = pypto.expand_clone(denom, [batch, BASIC_CHANNEL, 1])
            normed = (x_chunk - mean_b) / denom_b
            pypto.assemble(normed, [0, ch_off, 0], output)
        return output
    return kernel
```

forward：`reshape(B, C, -1)` → kernel → `reshape(x.shape)`
RMSNorm 同模式：3D `(B, features, spatial)`，只求 `sqrt(mean(x²) + eps)` 无需减均值。

## 模式要点
- `pypto.sum(dim=2)` 再 `pypto.sum(dim=0)` — 多轴归约必须分步
- `pypto.expand_clone(mean, [B, C, 1])` — 单轴广播，归约后恢复维度用于运算
- `set_vec_tile_shapes(1, 1, 16384)` — 3D，前两维小，最后维大 tile
