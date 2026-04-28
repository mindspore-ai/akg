import torch
import triton
import triton.language as tl


@triton.jit
def aikg_35_GroupNorm__kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    G: tl.constexpr,
    eps: tl.constexpr,
    CPG: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tasks = B * G
    total_elements = CPG * HW
    inv_count = 1.0 / total_elements

    tile_offsets = tl.arange(0, TILE_SIZE)

    for task_idx in range(pid, total_tasks, NUM_CORES):
        pid_b = task_idx // G
        pid_g = task_idx % G
        start_ch = pid_g * CPG

        sum_acc = 0.0
        sq_acc = 0.0

        for c in range(CPG):
            base = pid_b * C * HW + (start_ch + c) * HW
            for hw_start in range(0, HW, TILE_SIZE):
                data = tl.load(x_ptr + base + hw_start + tile_offsets)
                sum_acc += tl.sum(data)
                sq_acc += tl.sum(data * data)

        mean_val = sum_acc * inv_count
        var_val = sq_acc * inv_count - mean_val * mean_val
        inv_std = 1.0 / tl.sqrt(var_val + eps)

        for c in range(CPG):
            base = pid_b * C * HW + (start_ch + c) * HW
            for hw_start in range(0, HW, TILE_SIZE):
                data = tl.load(x_ptr + base + hw_start + tile_offsets)
                normalized = (data - mean_val) * inv_std
                tl.store(y_ptr + base + hw_start + tile_offsets, normalized)


def aikg_35_GroupNorm__triton_ascend_torch(x, eps=1e-5):
    B, C, H, W = x.shape
    G = 8
    assert C % G == 0
    CPG = C // G

    y = torch.empty_like(x)

    NUM_CORES = 40
    HW = H * W
    TILE_SIZE = 16384

    aikg_35_GroupNorm__kernel[(NUM_CORES,)](
        x, y, B, C, HW, G, eps,
        CPG=CPG, TILE_SIZE=TILE_SIZE, NUM_CORES=NUM_CORES,
    )

    return y
