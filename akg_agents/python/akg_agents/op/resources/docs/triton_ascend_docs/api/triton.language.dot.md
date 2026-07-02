### tl.dot(a, b, acc=None)
```python
result = tl.dot(a, b, acc=accumulator)
```
- **参数**:
  - `a`, `b`: 输入矩阵
  - `acc`: 累加器 (可选)
- **返回**: 矩阵乘法结果
- **用途**: 核心矩阵乘法操作
- **Ascend 约束**:
  - 不要传 `allow_tf32` 或 `input_precision`；这些是 CUDA 精度控制语义，Ascend 后端不支持。
  - `a` 和 `b` 通常应为 rank-2 block tensor，形状分别为 `(BLOCK_M, BLOCK_K)` 和 `(BLOCK_K, BLOCK_N)`；rank-3 表示 batched matmul。
  - K 维必须一致。真实 `K_total` 不是 tile 整数倍时，用 padded `BLOCK_K` 和 `offs_k < K_total` mask，masked load 的 `other` 使用 `0.0`。
  - `BLOCK_K` 通常取 cube/tensor-core 友好的大小，常见至少 16；不要使用 `(1, BLOCK_K) @ (BLOCK_K, 1)` 伪装矩阵化。
  - `acc += A * B` 不能替代 `tl.dot(A, B)`，它只做逐元素乘法/广播，不会沿 K 做矩阵乘归约。
