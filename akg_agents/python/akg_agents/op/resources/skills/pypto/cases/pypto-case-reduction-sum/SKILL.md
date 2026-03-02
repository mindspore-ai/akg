---
name: pypto-case-reduction-sum
description: "单轴归约示例：3D Sum reduction — 保持原始维度，最简 kernel"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "reduction"
---

# 单轴归约：Sum Reduction（3D）

最简单的模式——不需要 loop/view/assemble，kernel 只有 3 行。

```python
def create_sum_reduction_kernel(in_shape, out_shape):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor(in_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        output = pypto.tensor(list(out_shape), pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 16, 256)
        output[:] = pypto.sum(x, dim=1, keepdim=True)
        return output
    return kernel
```

forward：**保持原始维度，不降维**。

```python
def forward(self, x):
    assert x.dim() == 3
    assert tuple(x.shape) == (16, 256, 256)
    assert self.dim == 1
    x = x.contiguous()
    batch, _, dim2 = x.shape
    return create_sum_reduction_kernel(
        tuple(x.shape), (batch, 1, dim2)
    )(x)
```

## 模式要点
- **保持输入原始维度**，不 reshape → tile 参数个数 = input rank
- kernel 极简：`set_tile + sum + return`，无需 loop/view/assemble
- `pypto.amin` / `pypto.amax` 同理，只换 API
- mean = `sum * (1.0 / size)`（无内建 mean API）
- 对 `(16, 256, 256), dim=1` 的 3D 单轴归约，默认首选从 `(1, 16, 256)` 起步，再按 `32/64` 对照实测。
- 这里不要套用 loop 的“中段先试”习惯到 tile；该固定形状直接以 `(1, 16, 256)` 作为默认实现。
- 对该固定形状，`(1, 32, 256)` 与 `(1, 64, 256)` 不作为默认模板，仅作为对照候选。

## 强约束（reduction_over_a_dimension 系列）
- 本系列题目的 `get_init_inputs()` 返回值是**本次固定参数**（例如 `dim=1`）。
- 注释里的 `Example, change to desired dimension` 是题库说明，不是当前实现目标。
- 生成代码时：
  - 保留 `ModelNew.__init__(dim)` 签名；
  - `forward` 中 `assert self.dim == <固定值>`；
  - kernel 使用固定常量 `dim=<固定值>`，不要写 `if dim == ...` 分支；
  - 固定 dim 场景下，`create_*_kernel` 不再接收 `dim` 运行时参数。

反例（不要这样写）：
```python
def create_xxx_kernel(in_shape, out_shape, dim):
    ...
    output[:] = pypto.amin(x, dim=dim, keepdim=True)
```

正例（固定 dim 写死）：
```python
FIXED_DIM = 1
def create_xxx_kernel(in_shape, out_shape):
    ...
    output[:] = pypto.amin(x, dim=FIXED_DIM, keepdim=True)
```
