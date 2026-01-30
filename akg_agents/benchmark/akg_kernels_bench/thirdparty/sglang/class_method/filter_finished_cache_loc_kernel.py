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
#   filter_finished_cache_loc_kernel - 过滤已完成的缓存位置
# ============================================================================


@triton.jit
def filter_finished_cache_loc_kernel(
    out_cache_loc,
    tgt_cache_loc,
    accept_length,
    accept_length_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    accept_length_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    old_start = tl.sum(accept_length_all) + bid

    accept_length_filter_all = tl.load(
        accept_length_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(accept_length_filter_all)

    copy_len = tl.load(accept_length_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tgt_cache_loc, accept_length, accept_length_filter):
        batch_size = accept_length.shape[0]

        # Calculate output size
        out_size = accept_length_filter.sum().item()

        # Allocate output tensor
        out_cache_loc = torch.empty(out_size, dtype=tgt_cache_loc.dtype, device=tgt_cache_loc.device)

        # Calculate max copy length
        max_copy_len = accept_length_filter.max().item() if batch_size > 0 else 0

        # Find next power of 2
        bs_upper = triton.next_power_of_2(batch_size)
        num_verify_tokens_upper = triton.next_power_of_2(max(max_copy_len, 1))

        # Launch kernel
        grid = (batch_size,)
        filter_finished_cache_loc_kernel[grid](
            out_cache_loc,
            tgt_cache_loc,
            accept_length,
            accept_length_filter,
            bs_upper,
            num_verify_tokens_upper,
        )

        return out_cache_loc


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, tgt_cache_loc, accept_length, accept_length_filter):
        batch_size = accept_length.shape[0]

        # Calculate output size
        out_size = accept_length_filter.sum().item()

        # Allocate output tensor
        out_cache_loc = torch.empty(out_size, dtype=tgt_cache_loc.dtype, device=tgt_cache_loc.device)

        # Process each batch element
        for bid in range(batch_size):
            # Calculate old_start: where to read from tgt_cache_loc
            old_start = accept_length[:bid].sum().item() + bid

            # Calculate new_start: where to write to out_cache_loc
            new_start = accept_length_filter[:bid].sum().item()

            # Copy length
            copy_len = accept_length_filter[bid].item()

            if copy_len > 0:
                # Copy from tgt_cache_loc to out_cache_loc
                out_cache_loc[new_start:new_start + copy_len] = tgt_cache_loc[old_start:old_start + copy_len]

        return out_cache_loc


def get_inputs():
    # Example dimensions
    batch_size = 8
    dtype = torch.int32

    # accept_length: original accept lengths (e.g., 0 to 15)
    accept_length = torch.randint(0, 16, (batch_size,), dtype=dtype)

    # accept_length_filter: filtered accept lengths
    # Some elements are 0 (finished), others are accept_length[i] + 1
    # Simulate: randomly set some to 0 (finished)
    finished_mask = torch.rand(batch_size) < 0.3  # 30% chance of being finished
    accept_length_filter = accept_length.clone()
    accept_length_filter[finished_mask] = 0
    accept_length_filter[~finished_mask] = accept_length[~finished_mask] + 1

    # tgt_cache_loc: target cache locations
    # Size should be sum(accept_length) + batch_size (based on get_target_cache_loc output)
    tgt_cache_loc_size = (accept_length + 1).sum().item() + batch_size
    tgt_cache_loc = torch.randint(0, 1000, (tgt_cache_loc_size,), dtype=dtype)

    return [tgt_cache_loc, accept_length, accept_length_filter]


def get_init_inputs():
    return []
