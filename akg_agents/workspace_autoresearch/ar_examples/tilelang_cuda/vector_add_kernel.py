import tilelang
import tilelang.language as T
import torch
import torch.nn as nn


def vector_add_kernel(block_m, block_n):
    m = T.dynamic("m")
    n = T.dynamic("n")

    @T.prim_func
    def main(
        x: T.Tensor((m, n), "float16"),
        y: T.Tensor((m, n), "float16"),
        out: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m),
                      threads=128) as (bx, by):
            x_shared = T.alloc_shared((block_m, block_n), "float16")
            y_shared = T.alloc_shared((block_m, block_n), "float16")
            out_local = T.alloc_fragment((block_m, block_n), "float16")
            out_shared = T.alloc_shared((block_m, block_n), "float16")

            T.copy(x[by * block_m, bx * block_n], x_shared)
            T.copy(y[by * block_m, bx * block_n], y_shared)
            for i, j in T.Parallel(block_m, block_n):
                out_local[i, j] = x_shared[i, j] + y_shared[i, j]
            T.copy(out_local, out_shared)
            T.copy(out_shared, out[by * block_m, bx * block_n])

    return main


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        block_m = 32
        block_n = 256
        func = vector_add_kernel(block_m, block_n)
        self.kernel = tilelang.compile(func, out_idx=[-1], target="cuda")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x.contiguous(), y.contiguous())
