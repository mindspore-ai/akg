---
name: tilelang-ascend-example-quant-matmul
description: "量化矩阵乘法的 TileLang Ascend 实现示例。展示 Cube+Vector 融合编程：T.gemm_v0 int8 矩阵乘 + T.tile.cast 量化反量化 + workspace 跨核通信 + workspace_idx 声明。当生成量化/混合精度 matmul 类算子时可参考此示例。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# 量化矩阵乘法 — TileLang Ascend 实现示例（Cube+Vector 融合，Developer 模式）

**关键技术点**：
- `T.gemm_v0` int8 矩阵乘（accum_dtype="int32"）
- `T.tile.cast` int32→float32→float16 类型转换
- `workspace_idx=[4]` 声明 workspace 参数
- Cube 核写 workspace，Vector 核从 workspace 读取并反量化
- `AUTO_CV_COMBINE: True + AUTO_CV_SYNC: True` 自动 Cube+Vector 融合

```python
import tilelang as tl
import tilelang.language as T


@tl.jit(
    out_idx=[3],
    workspace_idx=[4],
    pass_configs={
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
)
def simple_quant_matmul(
    M, N, K, scale_size,
    block_M, block_N, block_K,
    in_dtype="int8", out_dtype="float16",
    accum_dtype="int32", scale_dtype="float32"
):
    VEC_NUM = 2
    CAST_MODE = "CAST_RINT"

    N_scale = N if scale_size == "N" else 1

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    block_M_2 = T.ceildiv(block_M, VEC_NUM)

    @T.prim_func
    def main(
            A: T.Tensor([M, K], in_dtype),
            B: T.Tensor([K, N], in_dtype),
            scale: T.Tensor([N_scale], scale_dtype),
            C: T.Tensor([M, N], out_dtype),
            workspace_1: T.Tensor([M, N], accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num

            A_L1 = T.alloc_L1([block_M, block_K], in_dtype)
            B_L1 = T.alloc_L1([block_K, block_N], in_dtype)

            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            c_ub = T.alloc_ub([block_M_2, block_N], accum_dtype)
            c_scale = T.alloc_ub([block_M_2, block_N], scale_dtype)
            c_out = T.alloc_ub([block_M_2, block_N], out_dtype)

            scale_ub = T.alloc_ub([block_N], scale_dtype)

            for bk in T.serial(k_num):
                T.copy(A[bm * block_M, bk * block_K], A_L1)
                T.copy(B[bk * block_K, bn * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))

            T.copy(C_L0, workspace_1[bm * block_M, bn * block_N])

            T.copy(workspace_1[bm * block_M + vid * block_M_2, bn * block_N], c_ub)

            if scale_size == "N":
                T.copy(scale[bn * block_N], scale_ub)
            else:
                T.copy(scale[0], scale_ub)
                T.tile.fill(scale_ub, scale_ub[0])

            if accum_dtype != scale_dtype:
                T.tile.cast(c_scale, c_ub, mode=CAST_MODE, count=block_M_2 * block_N)
            else:
                T.copy(c_ub, c_scale)

            for bm_v, bn_v in T.Parallel(block_M_2, block_N):
                c_scale[bm_v, bn_v] *= scale_ub[bn_v]

            if out_dtype != scale_dtype:
                T.tile.cast(c_out, c_scale, mode=CAST_MODE, count=block_M_2 * block_N)
            else:
                T.copy(c_scale, c_out)

            T.copy(c_out, C[bm * block_M + vid * block_M_2, bn * block_N])

    return main
```

**调用方式**：

```python
kernel = simple_quant_matmul(M, N, K, scale_size="N", block_M=128, block_N=256, block_K=64)
C = kernel(A_npu, B_npu, scale_npu)
```

**注意**：`workspace_idx` 中声明的参数不需要手动传入 tensor，编译器会自动分配。调用时只传 A、B、scale 即可。
