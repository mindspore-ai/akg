---
name: tilelang-ascend-example-matmul
description: "标准矩阵乘法的 TileLang Ascend Expert 模式实现示例。展示 Cube 核编程：L1/L0C 显式内存分配、T.gemm_v0 调用、T.Scope(\"C\") 核分离、T.barrier_all() 同步、K 维循环累加。当生成 matmul 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# 矩阵乘法 — TileLang Ascend 实现示例（Expert 模式）

**编程模式**：Expert（手动管理 L1/L0C 内存层级）

**关键技术点**：
- `T.alloc_L1` / `T.alloc_L0C` 显式内存分配
- `T.gemm_v0(A_L1, B_L1, C_L0C, init=(k==0))` Cube 矩阵乘
- `T.Scope("C")` Cube 核分离
- `T.barrier_all()` 核内同步
- K 维循环累加，首次迭代 `init=True` 清零 L0C

```python
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

**调用方式**：

```python
func = matmul(M, N, K, 128, 256, 64)
c = func(a, b)
```

**约束**：M、N 必须是 block_M、block_N 的整数倍。非整除场景需在 Python 层 zero-padding 后裁剪，或使用 Developer 模式 + `T.ceildiv`。
