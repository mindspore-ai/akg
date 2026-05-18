---
name: tilelang-ascend-example-reduction
description: "行归约（reduce_min）的 TileLang Ascend Expert 模式 pipeline 实现示例。当生成 reduce 类算子时可参考此示例的代码结构与双缓冲流水线模式。"
category: example
version: "2.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# 行归约（reduce_min）— TileLang Ascend Pipeline 实现示例（Expert 模式）

**编程模式**：Expert（手动 `T.Scope("V")` + `T.barrier_all()` + 双缓冲流水线）

**关键技术点**：
- `T.alloc_ub` UB 内存分配（带 `stages` 维度的双缓冲）
- `T.reduce_min` 归约操作（同理可用 `T.reduce_max` / `T.reduce_sum`）
- `T.Scope("V")` Vector 核分离
- `T.barrier_all()` 核内同步
- `VEC_NUM = 2` 双核并行，`sub_M` 子块切分
- `stages = 2` 双缓冲流水线：当前块计算与下一块数据搬运并行

```python
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[1], target="ascendc")
def reduce_min_pipeline(M, N, block_M, block_N, sub_M, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    stages = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            vec_proc = block_M // sub_M

            a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((stages, sub_M // VEC_NUM), dtype)

            with T.Scope("V"):
                T.barrier_all()

                T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + 0 * sub_M, by * block_N], a_ub[0, :, :])
                T.barrier_all()

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.barrier_all()
                        T.copy(A[bx * block_M + vid * sub_M // VEC_NUM + (mm + 1) * sub_M, by * block_N],
                               a_ub[nxt, :, :])
                        T.barrier_all()

                    T.barrier_all()

                    T.reduce_min(a_ub[cur, :, :], b_ub[cur, :], dim=-1)

                    T.barrier_all()

                    T.copy(b_ub[cur, :], B[bx * block_M + vid * sub_M // VEC_NUM + mm * sub_M])
                    T.barrier_all()

    return main
```

**调用方式**：

```python
func = reduce_min_pipeline(M, N, block_M, block_N, sub_M)
c = func(a)
```

**约束**：M 必须是 block_M 的整数倍，N 必须是 block_N 的整数倍。非整除场景需 padding 或使用 Developer 模式 + `T.ceildiv`。