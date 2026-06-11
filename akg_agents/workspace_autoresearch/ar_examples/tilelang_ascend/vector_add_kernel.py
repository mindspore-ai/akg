import tilelang
import tilelang.language as T
import torch
import torch.nn as nn


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def vector_add_kernel(block_m, block_n, dtype="float16"):
    m = T.symbolic("M")
    n = T.symbolic("N")
    m_num = T.ceildiv(m, block_m)
    n_num = T.ceildiv(n, block_n)
    vec_num = 2

    @T.prim_func
    def main(
        x: T.Tensor((m, n), dtype),
        y: T.Tensor((m, n), dtype),
        out: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a_ub = T.alloc_ub((block_m // vec_num, block_n), dtype)
            b_ub = T.alloc_ub((block_m // vec_num, block_n), dtype)
            c_ub = T.alloc_ub((block_m // vec_num, block_n), dtype)

            row = bx * block_m + vid * block_m // vec_num
            col = by * block_n
            T.copy(x[row, col], a_ub)
            T.copy(y[row, col], b_ub)
            for i, j in T.Parallel(block_m // vec_num, block_n):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
            T.copy(c_ub, out[row, col])

    return main


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        block_m = 128
        block_n = 256
        func = vector_add_kernel(block_m, block_n)
        self.kernel = tilelang.compile(
            func, out_idx=[-1], pass_configs=pass_configs, target="ascendc"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x.contiguous(), y.contiguous())
