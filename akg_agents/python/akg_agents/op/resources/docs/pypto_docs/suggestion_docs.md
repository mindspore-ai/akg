# PyPTO 编程要点

## 核心原则

- **静态 shape**：kernel 一切编译期确定。forward 负责 assert/reshape/contiguous。
- **运算符规则**：`+` `*` 标量任意位置；**`-` `/` tensor 必须在左**（`1.0 - x` crash → `x * (-1.0) + 1.0`）。函数调用第一参数必须 Tensor。
- **tile**：`prod(tile)` ≤ 16384，`auto_tiles` ≤ 8192。1D `(8192)`, 2D `(1, 16384)`, 3D `(1, 64, 256)`。
- **标量通过闭包**：float/int 参数通过工厂函数传入。标量之间用 Python 运算。
- **代码极简**：kernel 通常 6-10 行。不要过度工程。
- **禁用 API**：`pypto.where`、`pypto.clamp`。

## 维度选择

- Elementwise / 简单 Loss → 1D（**大矩阵 >10M 元素保持 2D + loop**）
- GroupNorm / InstanceNorm → 2D | BatchNorm / RMSNorm → 3D
- Batched matmul（同 batch）→ 3D 不需要 loop | 2D matmul → loop M 轴
- 三角/对称/对角矩阵乘法 = 标准 matmul

## 关键模式

- **条件分支**：`maximum(x, 0) + g(minimum(x, 0))`。`pypto.minimum` 可用。
- **方差**：`var = sq_sum * inv_count - mean * mean`
- **距离**：TripletMarginLoss 必须 `sqrt(sum_sq + eps)`
- **matmul 转置**：`pypto.matmul(a, b, DT_FP32, b_trans=True)`
- **模块名 `pypto`**，不是 `pyto`。**ModelNew.__init__** 签名与原始 Model 一致。
