---
name: pypto-case-matmul-2d
description: "模式 B 示例：2D 矩阵乘法 + M 维 loop 分块 + 尾部处理"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "matmul,loop,linear,bias_add"
---

# 模式 B：Matmul + Loop（含尾部处理）

```python
def ceil_div(a, b):
    return (a + b - 1) // b

def create_matmul_kernel(m, k, n):
    # 先在 loop_count 空间选中段，再反推 BASIC_BATCH
    # 当 loop 范围约为 1~128 时，默认先试 16/32
    TARGET_LOOP_COUNT = 16
    BASIC_BATCH = ceil_div(m, TARGET_LOOP_COUNT)

    full_iterations = m // BASIC_BATCH
    tail = m % BASIC_BATCH
    tail_offset = full_iterations * BASIC_BATCH

    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        a: pypto.Tensor((m, k), pypto.DT_FP32),
        b: pypto.Tensor((k, n), pypto.DT_FP32),
    ) -> pypto.Tensor((m, n), pypto.DT_FP32):
        pypto.set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)
        c = pypto.tensor([m, n], pypto.DT_FP32)

        for idx in pypto.loop(0, full_iterations, 1, name="LOOP_M", idx_name="idx"):
            offset = idx * BASIC_BATCH
            a_chunk = pypto.view(a, [BASIC_BATCH, k], [offset, 0])
            c_chunk = pypto.matmul(a_chunk, b, pypto.DT_FP32)
            pypto.assemble(c_chunk, [offset, 0], c)

        if tail > 0:
            a_tail = pypto.view(a, [tail, k], [tail_offset, 0])
            c_tail = pypto.matmul(a_tail, b, pypto.DT_FP32)
            pypto.assemble(c_tail, [tail_offset, 0], c)

        return c
    return kernel
```

forward：assert → contiguous → 读 shape → 调 kernel

**3D 输入 + 2D B**：forward 中计算 `nm = N * M`，`A.reshape(nm, K)` → 将 `nm` 传入工厂函数（不要分别传 N、M）：
```python
def forward(self, A, B):
    N, M, K = A.shape
    nm = N * M
    A_2d = A.reshape(nm, K)
    result_2d = create_matmul_kernel(nm, K, L)(A_2d, B)
    return result_2d.reshape(N, M, L)
```

## Matmul + Bias（Linear）两阶段写法

`linear = matmul + bias` 不要把 `add` 直接塞在 cube 阶段。`matmul` 是 cube op，`add/expand_clone` 是 vec op，必须显式切换 tile。

```python
def create_linear_kernel(m, k, n):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        x: pypto.Tensor((m, k), pypto.DT_FP32),
        w: pypto.Tensor((k, n), pypto.DT_FP32),
        b_row: pypto.Tensor((1, n), pypto.DT_FP32),   # forward 中 b.reshape(1, -1)
    ) -> pypto.Tensor((m, n), pypto.DT_FP32):
        # Phase 1: cube matmul
        pypto.set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)
        mm = pypto.tensor([m, n], pypto.DT_FP32)
        for idx in pypto.loop(0, full_iterations, 1, name="LOOP_M", idx_name="idx"):
            off = idx * BASIC_BATCH
            x_chunk = pypto.view(x, [BASIC_BATCH, k], [off, 0])
            y_chunk = pypto.matmul(x_chunk, w, pypto.DT_FP32)
            pypto.assemble(y_chunk, [off, 0], mm)

        # Phase 2: vec bias add
        pypto.set_vec_tile_shapes(1, n)
        b_full = pypto.expand_clone(b_row, [m, n])   # 单轴广播
        out = pypto.add(mm, b_full)
        return out
    return kernel
```

## 要点
- 禁止把 `BASIC_BATCH` 当固定答案；先定 `loop_count`，再反推 `BASIC_BATCH`。
- 当 `loop_count` 范围约为 `1~128` 且候选按 2 倍步长变化时，中段优先试 `16/32`（对数刻度中间，不是算术中点）。
- 例：`m=16384` 时，`loop=16/32` 对应 `BASIC_BATCH=1024/512`；再扩 `loop=8/64` 对应 `2048/256`。
- 避免两端极值：既不要盲目追求 `loop_count=1`，也不要默认用最小 batch 让 `loop_count` 接近最大。
- **view shape 必须是编译期常量**：`BASIC_BATCH`、`tail` 都是闭包常量
- **禁止** `min(BASIC_BATCH, m - offset)` 作为 view shape（offset 含 loop 变量 = 运行时值）
- `a_trans=True` / `b_trans=True` 支持转置，结构不变
- 三角/对称矩阵：直接标准 matmul
- M ≤ 128 时可不 loop：`c[:] = pypto.matmul(a, b, ...)`
- `matmul + elementwise` 混合时使用两阶段 tile：`set_cube_tile_shapes(...)` 后，进入 vec 阶段前再 `set_vec_tile_shapes(...)`。
