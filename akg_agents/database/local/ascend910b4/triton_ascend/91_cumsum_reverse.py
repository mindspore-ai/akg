import torch
import triton
import triton.language as tl


@triton.jit
def aikg_91_cumsum_reverse_kernel(
    x_ptr,
    y_ptr,
    B,
    N: tl.constexpr,
    stride_b,
    TILE_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_TILES: tl.constexpr = (N + TILE_SIZE - 1) // TILE_SIZE

    for b_idx in range(pid, B, NUM_CORES):
        carry = 0.0
        x_base = x_ptr + b_idx * stride_b
        y_base = y_ptr + b_idx * stride_b

        for tile_fwd in range(NUM_TILES):
            tile_idx = NUM_TILES - 1 - tile_fwd
            start = tile_idx * TILE_SIZE
            offsets = start + tl.arange(0, TILE_SIZE)
            mask = offsets < N
            data = tl.load(x_base + offsets, mask=mask, other=0.0)
            cs = tl.cumsum(data, axis=0, reverse=True)
            result = cs + carry
            tl.store(y_base + offsets, result, mask=mask)
            carry = carry + tl.sum(data)


def aikg_91_cumsum_reverse_triton_ascend_torch(x, dim=1):
    if not x.is_contiguous():
        x = x.contiguous()

    B, N = x.shape
    y = torch.empty_like(x)

    stride_b = x.stride(0)

    NUM_CORES = 40
    grid = (NUM_CORES,)

    aikg_91_cumsum_reverse_kernel[grid](
        x, y, B, N, stride_b,
        TILE_SIZE=4096, NUM_CORES=NUM_CORES,
    )

    return y
