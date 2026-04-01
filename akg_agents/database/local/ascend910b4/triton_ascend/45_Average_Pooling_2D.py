import torch
import triton
import triton.language as tl


@triton.jit
def aikg_45_Average_Pooling_2D_kernel(
    input_ptr,
    output_ptr,
    BC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride_val: tl.constexpr,
    BLOCK_W: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = BC * H_out
    inv_area = 1.0 / (kernel_size * kernel_size)
    out_local = tl.arange(0, BLOCK_W)
    out_mask = out_local < W_out

    for row_idx in range(pid, total_rows, NUM_CORES):
        bc_idx = row_idx // H_out
        h_out_idx = row_idx % H_out
        h_start = h_out_idx * stride_val
        in_base = bc_idx * (H * W)
        out_base = bc_idx * (H_out * W_out) + h_out_idx * W_out

        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

        for kh in range(kernel_size):
            h_in = h_start + kh
            row_base = in_base + h_in * W
            ptr = tl.make_block_ptr(
                base=input_ptr + row_base,
                shape=(W_out, kernel_size),
                strides=(stride_val, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_W, kernel_size),
                order=(1, 0)
            )
            tile = tl.load(ptr, boundary_check=(0,))
            acc += tl.sum(tile, axis=1)

        avg = acc * inv_area
        tl.store(output_ptr + out_base + out_local, avg, mask=out_mask)


def aikg_45_Average_Pooling_2D_triton_ascend_torch(x, kernel_size=3, stride=None, padding=0):
    if stride is None:
        stride = kernel_size

    B, C, H, W = x.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    output = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)

    NUM_CORES = 40
    BLOCK_W = 128
    BC = B * C

    x_flat = x.contiguous().view(BC, H, W)
    out_flat = output.view(BC, H_out, W_out)

    aikg_45_Average_Pooling_2D_kernel[(NUM_CORES,)](
        x_flat, out_flat,
        BC, H, W, H_out, W_out,
        kernel_size, stride,
        BLOCK_W=BLOCK_W, NUM_CORES=NUM_CORES,
    )

    return output
