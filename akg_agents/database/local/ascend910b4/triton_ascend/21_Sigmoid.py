import torch
import triton
import triton.language as tl


@triton.jit
def aikg_21_Sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)

    for block_start in range(pid * BLOCK_SIZE, n_elements, NUM_CORES * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x_tile = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        neg_x = -x_tile
        exp_neg_x = tl.exp(neg_x)
        y_tile = 1.0 / (1.0 + exp_neg_x)
        tl.store(y_ptr + offsets, y_tile, mask=mask)


def aikg_21_Sigmoid_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()

    y = torch.empty_like(x)
    n_elements = x.numel()

    NUM_CORES = 40
    grid = (NUM_CORES,)

    aikg_21_Sigmoid_kernel[grid](
        x, y, n_elements,
        BLOCK_SIZE=4096, NUM_CORES=NUM_CORES,
    )

    return y
