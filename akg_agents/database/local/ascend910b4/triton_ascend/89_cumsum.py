import torch
import triton
import triton.language as tl


@triton.jit
def aikg_89_cumsum_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    stride_b: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N

    for b_idx in range(pid, B, NUM_CORES):
        data = tl.load(x_ptr + b_idx * stride_b + offsets, mask=mask, other=0.0)
        cs = tl.cumsum(data, axis=0)
        tl.store(y_ptr + b_idx * stride_b + offsets, cs, mask=mask)


def aikg_89_cumsum_triton_ascend_torch(x, dim=1):
    if not x.is_contiguous():
        x = x.contiguous()

    B, N = x.shape
    y = torch.empty_like(x)

    stride_b = x.stride(0)

    NUM_CORES = 40
    TILE_SIZE = 4096

    aikg_89_cumsum_kernel[(NUM_CORES,)](
        x, y, B, N, stride_b,
        TILE_SIZE=TILE_SIZE, NUM_CORES=NUM_CORES,
    )

    return y
