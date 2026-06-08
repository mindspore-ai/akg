import tilelang
import tilelang.language as T
import torch
import torch.nn as nn


def relu_kernel(block_M, block_N):
    M = T.dynamic("m")
    N = T.dynamic("n")

    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float16"),
        Y: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), "float16")
            Y_local = T.alloc_fragment((block_M, block_N), "float16")
            Y_shared = T.alloc_shared((block_M, block_N), "float16")

            T.copy(X[by * block_M, bx * block_N], X_shared)
            T.copy(X_shared, Y_local)
            for i, j in T.Parallel(block_M, block_N):
                Y_local[i, j] = T.max(Y_local[i, j], 0)
            T.copy(Y_local, Y_shared)
            T.copy(Y_shared, Y[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        block_M = 128
        block_N = 256
        func = relu_kernel(block_M, block_N)
        self.kernel = tilelang.compile(
            func, out_idx=[-1], target="cuda"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return self.kernel(x)
