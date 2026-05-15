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
#   align_evict_mask_to_page_size_kernel - 将驱逐掩码对齐到页面大小
# ============================================================================


@triton.jit
def align_evict_mask_to_page_size_kernel(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len

    # Calculate the range where we need to set False
    start_clamped = tl.maximum(start, 0)
    end = tl.minimum(start + page_size, num_draft_tokens)

    # Create a mask for the range [start_clamped, end)
    range_mask = (t_range >= start_clamped) & (t_range < end) & io_mask

    # Store False for the positions in the range
    tl.store(
        evict_mask + bid * num_draft_tokens + t_range,
        False,
        mask=range_mask,
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, seq_lens, evict_mask, page_size):
        batch_size = seq_lens.shape[0]
        num_draft_tokens = evict_mask.shape[1]

        # BLOCK_SIZE should be >= num_draft_tokens
        BLOCK_SIZE = triton.next_power_of_2(num_draft_tokens)

        # Launch kernel
        grid = (batch_size,)
        align_evict_mask_to_page_size_kernel[grid](
            seq_lens,
            evict_mask,
            page_size,
            num_draft_tokens,
            BLOCK_SIZE,
        )

        return evict_mask


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, seq_lens, evict_mask, page_size):
        batch_size = seq_lens.shape[0]
        num_draft_tokens = evict_mask.shape[1]

        # Clone to avoid modifying input
        result = evict_mask.clone()

        for bid in range(batch_size):
            seq_len = seq_lens[bid].item()
            mask_row = result[bid]

            # Count number of True and False values
            num_trues = mask_row.sum().item()
            num_false = num_draft_tokens - num_trues

            # Calculate start position for alignment
            start = (seq_len + num_false - 1) // page_size * page_size - seq_len

            # Set False in the alignment range
            start_idx = max(start, 0)
            end_idx = min(start + page_size, num_draft_tokens)

            for i in range(start_idx, end_idx):
                result[bid, i] = False

        return result


def get_inputs():
    # Example dimensions
    batch_size = 8
    num_draft_tokens = 32
    page_size = 8
    dtype = torch.int32

    # seq_lens: sequence lengths for each batch element
    seq_lens = torch.randint(50, 200, (batch_size,), dtype=dtype)

    # evict_mask: boolean mask indicating which tokens to evict
    evict_mask = torch.randint(0, 2, (batch_size, num_draft_tokens), dtype=torch.bool)

    return [seq_lens, evict_mask, page_size]


def get_init_inputs():
    return []
