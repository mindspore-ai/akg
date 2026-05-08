import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def relu(block_M, block_N, dtype="float16"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            x_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            y_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(
                X[bx * block_M + vid * block_M // VEC_NUM, by * block_N],
                x_ub,
            )
            T.tile.relu(y_ub, x_ub)
            T.copy(
                y_ub,
                Y[bx * block_M + vid * block_M // VEC_NUM, by * block_N],
            )

    return main


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        block_M = 128
        block_N = 256
        func = relu(block_M, block_N, dtype="float16")
        self.kernel = tilelang.compile(
            func, out_idx=[-1], pass_configs=pass_configs, target="ascendc"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return self.kernel(x)

