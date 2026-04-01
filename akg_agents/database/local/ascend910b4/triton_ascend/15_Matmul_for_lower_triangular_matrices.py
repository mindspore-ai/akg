import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32, 'K_TILE': 256}),
        triton.Config({'BLOCK_SIZE': 64, 'K_TILE': 128}),
        triton.Config({'BLOCK_SIZE': 128, 'K_TILE': 64}),
    ],
    key=['N'],
)
@triton.jit
def aikg_15_Matmul_for_lower_triangular_matrices_kernel(
    A_ptr, B_ptr, C_ptr,
    N: tl.constexpr,
    stride_A_m, stride_A_n,
    stride_B_m, stride_B_n,
    stride_C_m, stride_C_n,
    BLOCK_SIZE: tl.constexpr,
    K_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_cols = tl.cdiv(N, BLOCK_SIZE)

    block_row = pid // grid_cols
    block_col = pid % grid_cols

    if block_col <= block_row:
        accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

        for k_outer in range(0, tl.cdiv(N, K_TILE)):
            a_offsets_m = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            a_offsets_n = k_outer * K_TILE + tl.arange(0, K_TILE)
            a_mask = (a_offsets_m[:, None] < N) & (a_offsets_n[None, :] < N)
            a_tile = tl.load(
                A_ptr + a_offsets_m[:, None] * stride_A_m + a_offsets_n[None, :] * stride_A_n,
                mask=a_mask, other=0.0,
            )

            b_offsets_m = k_outer * K_TILE + tl.arange(0, K_TILE)
            b_offsets_n = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            b_mask = (b_offsets_m[:, None] < N) & (b_offsets_n[None, :] < N)
            b_tile = tl.load(
                B_ptr + b_offsets_m[:, None] * stride_B_m + b_offsets_n[None, :] * stride_B_n,
                mask=b_mask, other=0.0,
            )

            accumulator += tl.dot(a_tile, b_tile)

        c_offsets_m = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        c_offsets_n = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        c_mask = (c_offsets_m[:, None] < N) & (c_offsets_n[None, :] < N)
        tl.store(
            C_ptr + c_offsets_m[:, None] * stride_C_m + c_offsets_n[None, :] * stride_C_n,
            accumulator, mask=c_mask,
        )


def aikg_15_Matmul_for_lower_triangular_matrices_triton_ascend_torch(A, B):
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    N = A.shape[0]
    C = torch.zeros((N, N), dtype=torch.float32, device=A.device)

    def grid(meta):
        gc = triton.cdiv(N, meta['BLOCK_SIZE'])
        return (gc * gc,)

    aikg_15_Matmul_for_lower_triangular_matrices_kernel[grid](
        A, B, C, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C
