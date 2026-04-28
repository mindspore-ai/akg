import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def aikg_44_Average_Pooling_1D_kernel(
    x_ptr,
    y_ptr,
    BC: tl.constexpr,
    L_padded: tl.constexpr,
    L_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride_val: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_k = 1.0 / kernel_size
    out_offs = tl.arange(0, BLOCK_OUT)
    out_mask = out_offs < L_out

    for bc_idx in range(pid, BC, NUM_CORES):
        x_base = bc_idx * L_padded
        y_base = bc_idx * L_out

        ptr = tl.make_block_ptr(
            base=x_ptr + x_base,
            shape=(L_out, kernel_size),
            strides=(stride_val, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_OUT, kernel_size),
            order=(1, 0)
        )
        tile = tl.load(ptr, boundary_check=(0,))
        row_sum = tl.sum(tile, axis=1)
        avg = row_sum * inv_k
        tl.store(y_ptr + y_base + out_offs, avg, mask=out_mask)


def aikg_44_Average_Pooling_1D_triton_ascend_torch(x):
    B, C, L_in = x.shape

    kernel_size = 4
    stride = 2
    padding = 1

    L_out = (L_in + 2 * padding - kernel_size) // stride + 1
    y = torch.empty((B, C, L_out), dtype=x.dtype, device=x.device)

    NUM_CORES = 40
    BC = B * C
    L_padded = L_in + 2 * padding

    BLOCK_OUT = 1
    while BLOCK_OUT < L_out:
        BLOCK_OUT *= 2

    x_padded = F.pad(x, (padding, padding), mode='constant', value=0.0)
    x_flat = x_padded.contiguous().view(BC, L_padded)
    y_flat = y.view(BC, L_out)

    aikg_44_Average_Pooling_1D_kernel[(NUM_CORES,)](
        x_flat, y_flat, BC, L_padded, L_out, kernel_size, stride,
        NUM_CORES=NUM_CORES, BLOCK_OUT=BLOCK_OUT,
    )

    return y
