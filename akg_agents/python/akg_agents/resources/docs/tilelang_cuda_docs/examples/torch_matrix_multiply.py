import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def square_matrix_multiply(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
            A: T.Tensor((M, K), "float32"),
            B: T.Tensor((K, N), "float32"),
            C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 内存分配
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            # 清零
            T.clear(C_local)

            # K 维度循环
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # 数据加载
                T.copy(A[by * block_M, ko * block_K], A_shared)

                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # 计算
                T.gemm(A_shared, B_shared, C_local)

            # 结果写回
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

def square_matrix_multiply_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.size(0)
    block_M = 128
    block_N = 128
    block_K = 32
    # 输出矩阵
    C = torch.empty_like(A)

    kernel = square_matrix_multiply(N, N, N, block_M, block_N, block_K)
    kernel(A, B, C)
    return C