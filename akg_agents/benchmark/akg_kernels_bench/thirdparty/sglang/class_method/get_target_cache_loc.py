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
#   get_target_cache_loc_kernel - 获取目标缓存位置
# ============================================================================


@triton.jit
def get_target_cache_loc_kernel(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(accept_length + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_slots
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, accept_length, to_free_num_slots, out_cache_loc):
        batch_size = accept_length.shape[0]
        num_verify_tokens = out_cache_loc.shape[1]

        # Calculate output sizes
        tgt_cache_loc_size = (accept_length + 1).sum().item() + batch_size
        to_free_slots_size = to_free_num_slots.sum().item()

        # Allocate output tensors
        tgt_cache_loc = torch.empty(tgt_cache_loc_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
        to_free_slots = torch.empty(to_free_slots_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)

        # Find next power of 2 for bs_upper and num_verify_tokens_upper
        bs_upper = triton.next_power_of_2(batch_size)
        num_verify_tokens_upper = triton.next_power_of_2(num_verify_tokens)

        # Launch kernel
        grid = (batch_size,)
        get_target_cache_loc_kernel[grid](
            tgt_cache_loc,
            to_free_slots,
            accept_length,
            to_free_num_slots,
            out_cache_loc,
            num_verify_tokens,
            num_verify_tokens_upper,
            bs_upper,
        )

        return tgt_cache_loc, to_free_slots


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, accept_length, to_free_num_slots, out_cache_loc):
        batch_size = accept_length.shape[0]
        num_verify_tokens = out_cache_loc.shape[1]

        # Calculate output sizes
        tgt_cache_loc_size = (accept_length + 1).sum().item() + batch_size
        to_free_slots_size = to_free_num_slots.sum().item()

        # Allocate output tensors
        tgt_cache_loc = torch.empty(tgt_cache_loc_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
        to_free_slots = torch.empty(to_free_slots_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)

        # Process each batch element
        for bid in range(batch_size):
            # Part 1: Copy first part to tgt_cache_loc
            # tgt_cache_loc_start = sum(accept_length[:bid]) + bid
            tgt_start = accept_length[:bid].sum().item() + bid
            copy_len = accept_length[bid].item() + 1
            tgt_cache_loc[tgt_start:tgt_start + copy_len] = out_cache_loc[bid, :copy_len]

            # Part 2: Copy second part to to_free_slots
            # to_free_slots_start = sum(to_free_num_slots[:bid])
            free_start = to_free_num_slots[:bid].sum().item()
            free_len = to_free_num_slots[bid].item()
            if free_len > 0:
                start_idx = num_verify_tokens - free_len
                to_free_slots[free_start:free_start + free_len] = out_cache_loc[bid, start_idx:start_idx + free_len]

        return tgt_cache_loc, to_free_slots


def get_inputs():
    # Example dimensions
    batch_size = 8
    num_verify_tokens = 16
    dtype = torch.int32

    # accept_length: number of accepted tokens per batch element (0 to num_verify_tokens-2)
    accept_length = torch.randint(0, num_verify_tokens - 2, (batch_size,), dtype=dtype)

    # to_free_num_slots: number of slots to free per batch element (0 to num_verify_tokens//2)
    to_free_num_slots = torch.randint(0, num_verify_tokens // 2, (batch_size,), dtype=dtype)

    # out_cache_loc: cache locations for each batch element
    out_cache_loc = torch.randint(0, 1000, (batch_size, num_verify_tokens), dtype=dtype)

    return [accept_length, to_free_num_slots, out_cache_loc]


def get_init_inputs():
    return []
