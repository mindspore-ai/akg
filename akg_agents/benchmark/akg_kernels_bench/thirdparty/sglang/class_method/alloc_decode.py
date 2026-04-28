import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/mem_cache/radix_cache.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   内部使用,用于解码阶段的内存分配
# Triton Kernel:
#   alloc_decode_kernel - 为解码阶段分配KV cache页面
# ============================================================================


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, seq_lens, last_loc, free_page):
        batch_size = seq_lens.shape[0]
        page_size = 16

        out_indices = torch.empty(batch_size, dtype=torch.int32, device=seq_lens.device)

        # Find the smallest power of 2 that is >= batch_size
        bs_upper = 1
        while bs_upper < batch_size:
            bs_upper *= 2

        grid = (batch_size,)
        alloc_decode_kernel[grid](
            seq_lens,
            last_loc,
            free_page,
            out_indices,
            bs_upper,
            page_size,
        )

        return out_indices


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, seq_lens, last_loc, free_page):
        """Pure PyTorch implementation for verification"""
        batch_size = seq_lens.shape[0]
        page_size = 16

        out_indices = torch.empty(batch_size, dtype=torch.int32, device=seq_lens.device)

        # Calculate pre_lens (for decode, pre_len = seq_len - 1)
        pre_lens = seq_lens - 1

        # Calculate page allocation
        num_pages_before = (pre_lens + page_size - 1) // page_size
        num_pages_after = (seq_lens + page_size - 1) // page_size
        num_new_pages = num_pages_after - num_pages_before

        # Calculate new page start location for each request
        new_page_start_locs = torch.zeros(batch_size, dtype=torch.int32, device=seq_lens.device)
        for i in range(1, batch_size):
            new_page_start_locs[i] = new_page_start_locs[i-1] + num_new_pages[i-1]

        # Process each request
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            pre_len = seq_len - 1
            new_page_start = new_page_start_locs[i].item()

            # Check if we need a new page
            num_page_self = num_pages_after[i].item() - num_pages_before[i].item()

            if num_page_self == 0:
                # No new page needed, use last_loc + 1
                last_loc_val = last_loc[i].item()
                out_indices[i] = last_loc_val + 1
            else:
                # Need a new page
                page_idx = new_page_start
                page_start = free_page[page_idx].item()
                out_indices[i] = page_start * page_size

        return out_indices


def get_inputs():
    # Example dimensions
    batch_size = 8
    page_size = 16
    dtype = torch.int32

    # Generate test inputs (不使用 device='cuda')
    # seq_lens should be > 0
    seq_lens = torch.randint(10, 100, (batch_size,), dtype=dtype)

    # last_loc should point to the last token position in the last page
    last_loc = torch.empty(batch_size, dtype=dtype)
    for i in range(batch_size):
        seq_len = seq_lens[i].item()
        pre_len = seq_len - 1  # decode: pre_len = seq_len - 1
        if pre_len > 0:
            last_page_idx = (pre_len - 1) // page_size
            last_loc[i] = last_page_idx * page_size + (pre_len - 1) % page_size
        else:
            last_loc[i] = 0

    # Calculate total new pages needed
    pre_lens = seq_lens - 1
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_pages_after = (seq_lens + page_size - 1) // page_size
    total_new_pages = (num_pages_after - num_pages_before).sum().item()

    # Generate free page indices
    free_page = torch.arange(100, 100 + total_new_pages, dtype=dtype)

    return [seq_lens, last_loc, free_page]


def get_init_inputs():
    return []
