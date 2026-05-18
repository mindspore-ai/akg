---
name: tilelang-ascend-example-elementwise
description: "Broadcast（行扩展）的 TileLang Ascend 实现示例。展示纯 Vector 核编程：T.alloc_ub UB 内存分配、T.tile.broadcast 向量广播、T.Scope(\"V\") Vector 域、T.barrier_all() 核内同步。当生成 elementwise 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# Broadcast（行扩展）— TileLang Ascend 实现示例

**编程模式**：Expert（纯 Vector 核，手动管理 UB 内存层级）

**关键技术点**：
- `T.alloc_ub` UB 内存分配（Vector 核专用）
- `T.tile.broadcast` 向量广播（将 1×N 扩展为 sub_block_M×N）
- `T.Scope("V")` Vector 域标注
- `T.barrier_all()` 核内 Cube/Vector 同步
- `vid` 子向量 ID，将 block_M 拆分给 2 个 Vector 核并行处理

```python
import tilelang
from tilelang import language as T


@tilelang.jit(out_idx=[1])
def broadcast(M, N, block_M, dtype="float"):
    m_num = M // block_M
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([1, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, N), dtype)
            b_ub = T.alloc_ub((sub_block_M, N), dtype)

            row_base = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                T.copy(A[0, :], a_ub)

                T.barrier_all()
                T.tile.broadcast(b_ub, a_ub)
                T.barrier_all()

                T.copy(b_ub, B[row_base : row_base + sub_block_M, :])

    return main
```

**elementwise 类算子通用模式**：
1. 数据从 Global → UB（`T.copy`）
2. 在 UB 上执行逐元素计算（`T.tile.add/mul/sub/div/exp` 或 `T.Parallel` 循环）
3. 结果从 UB → Global（`T.copy`）
4. 纯 Vector 操作无需 `T.Scope("C")` / `T.gemm_v0`
