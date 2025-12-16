import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/speculative/spec_utils.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   内部使用,用于推测解码
# Triton Kernel:
#   create_extend_after_decode_spec_info_kernel - 在解码后创建扩展的推测信息
# ============================================================================


@triton.jit
def create_extend_after_decode_spec_info_kernel(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    accept_length = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_length
    tl.store(positions_ptr + offsets, seq_length - accept_length + offsets, mask)

    accept_len_cumsum += accept_length - 1
    verified_id_data = tl.load(verified_id + accept_len_cumsum)
    tl.store(new_verified_id + pid, verified_id_data)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, verified_id, seq_lens, accept_lens):
        batch_size = seq_lens.shape[0]

        # Calculate output sizes
        positions_size = accept_lens.sum().item()

        # Allocate output tensors
        positions = torch.empty(positions_size, dtype=seq_lens.dtype, device=seq_lens.device)
        new_verified_id = torch.empty(batch_size, dtype=verified_id.dtype, device=verified_id.device)

        # Find next power of 2
        bs_upper = triton.next_power_of_2(batch_size)

        # Launch kernel
        grid = (batch_size,)
        create_extend_after_decode_spec_info_kernel[grid](
            verified_id,
            seq_lens,
            accept_lens,
            positions,
            new_verified_id,
            bs_upper,
        )

        return positions, new_verified_id


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, verified_id, seq_lens, accept_lens):
        batch_size = seq_lens.shape[0]

        # Calculate output sizes
        positions_size = accept_lens.sum().item()

        # Allocate output tensors
        positions = torch.empty(positions_size, dtype=seq_lens.dtype, device=seq_lens.device)
        new_verified_id = torch.empty(batch_size, dtype=verified_id.dtype, device=verified_id.device)

        # Process each batch element
        verified_id_idx = 0
        positions_idx = 0

        for bid in range(batch_size):
            seq_length = seq_lens[bid].item()
            accept_length = accept_lens[bid].item()

            # Write positions: [seq_length - accept_length, ..., seq_length - 1]
            if accept_length > 0:
                start_pos = seq_length - accept_length
                positions[positions_idx:positions_idx + accept_length] = torch.arange(
                    start_pos, seq_length, dtype=seq_lens.dtype, device=seq_lens.device
                )

                # Read the last verified_id for this batch element
                new_verified_id[bid] = verified_id[verified_id_idx + accept_length - 1]

                verified_id_idx += accept_length
                positions_idx += accept_length
            else:
                # If accept_length is 0, we might have a special case
                # But based on the kernel, it still reads verified_id at cumsum + accept_length - 1
                # which would be cumsum - 1. This might be an edge case.
                # For now, we'll handle it by setting a default value
                new_verified_id[bid] = 0

        return positions, new_verified_id


def get_inputs():
    # Example dimensions
    batch_size = 8
    dtype = torch.int32

    # accept_lens: number of accepted tokens per batch element (1 to 10)
    accept_lens = torch.randint(1, 11, (batch_size,), dtype=dtype)

    # seq_lens: current sequence lengths (must be >= accept_lens)
    seq_lens = torch.randint(50, 200, (batch_size,), dtype=dtype)
    # Ensure seq_lens >= accept_lens
    seq_lens = torch.maximum(seq_lens, accept_lens)

    # verified_id: verified token IDs (total length = sum(accept_lens))
    verified_id_size = accept_lens.sum().item()
    verified_id = torch.randint(0, 50000, (verified_id_size,), dtype=dtype)

    return [verified_id, seq_lens, accept_lens]


def get_init_inputs():
    return []
