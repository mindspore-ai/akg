import torch
import triton
import triton.language as tl


@triton.jit
def aikg_8_Matmul_with_irregular_shapes__kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        start_m = block_m * BLOCK_M
        start_n = block_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                offsets=(start_m, k),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0)
            )
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(k, start_n),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0)
            )

            a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
            b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
            accumulator += tl.dot(a_tile, b_tile)

        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(start_m, start_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


def aikg_8_Matmul_with_irregular_shapes__triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"K mismatch: {K} != {K2}"

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    num_cores = 20
    grid = (num_cores,)

    aikg_8_Matmul_with_irregular_shapes__kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        num_cores=num_cores,
        BLOCK_M=128, BLOCK_K=64, BLOCK_N=128,
    )

    return C
