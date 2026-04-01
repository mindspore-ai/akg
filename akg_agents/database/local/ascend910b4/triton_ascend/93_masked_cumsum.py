import torch
import triton
import triton.language as tl


@triton.jit
def aikg_93_masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, TILE_SIZE)
    vmask = offsets < N

    for b_idx in range(pid, B, NUM_CORES):
        x_tile = tl.load(x_ptr + b_idx * N + offsets, mask=vmask, other=0.0)
        m_tile = tl.load(mask_ptr + b_idx * N + offsets, mask=vmask, other=False)
        x_masked = x_tile * m_tile.to(tl.float32)
        cs = tl.cumsum(x_masked, axis=0)
        tl.store(output_ptr + b_idx * N + offsets, cs, mask=vmask)


def aikg_93_masked_cumsum_triton_ascend_torch(x, mask, dim=1):
    if not x.is_contiguous():
        x = x.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()

    B = x.shape[0]
    N = x.shape[1]

    output = torch.empty_like(x)

    NUM_CORES = 40
    TILE_SIZE = 4096

    aikg_93_masked_cumsum_kernel[(NUM_CORES,)](
        x, mask, output, B, N,
        TILE_SIZE=TILE_SIZE, NUM_CORES=NUM_CORES,
    )

    return output
