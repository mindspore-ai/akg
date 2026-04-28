import torch
import triton
import triton.language as tl


@triton.jit
def aikg_18_Matmul_with_transposed_both_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_ak, stride_am,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_cores: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """C = A.T @ B.T, A: [K, M], B: [N, K], C: [M, N]"""
    pid = tl.program_id(0)

    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    total_blocks = num_blocks_m * num_blocks_n

    for block_idx in range(pid, total_blocks, num_cores):
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n

        if M >= N:
            task_m_idx, task_n_idx = tl.swizzle2d(block_m, block_n, num_blocks_m, num_blocks_n, GROUP_SIZE)
        else:
            size_gj = GROUP_SIZE * num_blocks_m
            group_id = block_idx // size_gj
            off_n = group_id * GROUP_SIZE
            cur_size_g = min(num_blocks_n - off_n, GROUP_SIZE)
            local_ij = block_idx % size_gj
            task_m_idx = local_ij // cur_size_g
            task_n_idx = off_n + local_ij % cur_size_g

        start_m = task_m_idx * BLOCK_M
        start_n = task_n_idx * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_outer in range(0, K, BLOCK_K):
            # B[N,K]: load [BLOCK_N, BLOCK_K] at (start_n, k_outer)
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr,
                shape=(N, K),
                strides=(stride_bn, stride_bk),
                offsets=(start_n, k_outer),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0)
            )
            # A[K,M]: load [BLOCK_K, BLOCK_M] at (k_outer, start_m)
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(K, M),
                strides=(stride_ak, stride_am),
                offsets=(k_outer, start_m),
                block_shape=(BLOCK_K, BLOCK_M),
                order=(1, 0)
            )

            b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
            a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))

            # b_tile: [BLOCK_N, BLOCK_K] @ a_tile: [BLOCK_K, BLOCK_M] -> [BLOCK_N, BLOCK_M]
            gemm_result = tl.dot(b_tile, a_tile)
            accumulator += tl.trans(gemm_result)

        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(start_m, start_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


def aikg_18_Matmul_with_transposed_both_triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A.T @ B.T, A: [K, M], B: [N, K]"""
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    K, M = A.shape
    N, K2 = B.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    num_cores = 20
    grid = (num_cores,)

    aikg_18_Matmul_with_transposed_both_kernel[grid](
        A, B, C,
        M, K, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_K=256, BLOCK_N=64, num_cores=num_cores, GROUP_SIZE=4,
    )

    return C
