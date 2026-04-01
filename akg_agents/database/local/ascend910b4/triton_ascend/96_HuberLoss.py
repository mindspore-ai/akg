import torch
import triton
import triton.language as tl


@triton.jit
def aikg_96_HuberLoss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    N: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = 0.0

    for block_start in range(pid * BLOCK_SIZE, N, NUM_CORES * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        pred = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

        diff = pred - target
        abs_diff = tl.abs(diff)

        branch1 = 0.5 * diff * diff / beta
        branch2 = abs_diff - 0.5 * beta
        result = tl.where(abs_diff < beta, branch1, branch2)
        acc += tl.sum(result)

    tl.atomic_add(output_ptr, acc)


def aikg_96_HuberLoss_triton_ascend_torch(predictions, targets, beta=1.0):
    assert predictions.device == targets.device

    predictions_flat = predictions.contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1)
    N = predictions_flat.numel()

    NUM_CORES = 40
    BLOCK_SIZE = 4096

    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)

    aikg_96_HuberLoss_kernel[(NUM_CORES,)](
        predictions_flat, targets_flat, output,
        N, beta,
        BLOCK_SIZE=BLOCK_SIZE, NUM_CORES=NUM_CORES,
    )

    loss = output[0] / N
    return loss
