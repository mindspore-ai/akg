---
name: tilelang-ascend-example-reduction-sum
description: "列归约（reduce_sum）的 TileLang Ascend 实现示例。当生成 reduce-x 类算子时可参考此示例的代码结构与双缓冲流水线模式。"
category: example
version: "2.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# 列归约（reduce_sum）

```python
import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def reduce_x_kernel(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    sub_N = block_N // VEC_NUM

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((N), dtype),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub([block_M, sub_N], dtype)
            sum_ub = T.alloc_ub([sub_N], dtype)

            T.tile.fill(sum_ub, 0.0)

            for m_start in T.serial(m_num):
                T.copy(
                    A[m_start * block_M : (m_start + 1) * block_M,
                      cid * block_N + vid * sub_N : cid * block_N + (vid + 1) * sub_N],
                    a_ub
                )
                T.reduce_sum(a_ub, sum_ub, dim=0, clear=False)
            
            T.copy(
                sum_ub,
                B[cid * block_N + vid * sub_N : cid * block_N + (vid + 1) * sub_N]
            )

    return main

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        M, N = x.shape

        block_M = ...
        block_N = ...

        func = reduce_x_kernel(M, N, block_M, block_N, "float32")
        return func(x)
```