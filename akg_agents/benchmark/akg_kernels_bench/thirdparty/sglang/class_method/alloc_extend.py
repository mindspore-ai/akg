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
#   内部使用,用于扩展阶段的内存分配
# Triton Kernel:
#   alloc_extend_kernel - 为扩展阶段分配KV cache页面
# ============================================================================


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    offset_many_page = tl.arange(0, max_num_extend_tokens)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,
    )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, pre_lens, seq_lens, last_loc, free_page):
        batch_size = pre_lens.shape[0]
        page_size = 16
        max_extend_len = (seq_lens - pre_lens).max().item()
        max_num_extend_tokens = ((max_extend_len + page_size - 1) // page_size) * page_size

        # Calculate total extend tokens
        extend_lens = seq_lens - pre_lens
        total_extend_tokens = extend_lens.sum().item()

        out_indices = torch.empty(total_extend_tokens, dtype=torch.int32, device=pre_lens.device)

        # Find the smallest power of 2 that is >= batch_size
        bs_upper = 1
        while bs_upper < batch_size:
            bs_upper *= 2

        grid = (batch_size,)
        alloc_extend_kernel[grid](
            pre_lens,
            seq_lens,
            last_loc,
            free_page,
            out_indices,
            bs_upper,
            page_size,
            max_num_extend_tokens,
        )

        return out_indices


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, pre_lens, seq_lens, last_loc, free_page):
        """Pure PyTorch implementation for verification"""
        batch_size = pre_lens.shape[0]
        page_size = 16

        extend_lens = seq_lens - pre_lens
        total_extend_tokens = extend_lens.sum().item()

        out_indices = torch.empty(total_extend_tokens, dtype=torch.int32, device=pre_lens.device)

        # Calculate output start location for each request
        output_start_locs = torch.zeros(batch_size, dtype=torch.int32, device=pre_lens.device)
        for i in range(1, batch_size):
            output_start_locs[i] = output_start_locs[i-1] + extend_lens[i-1]

        # Calculate page allocation
        num_pages_before = (pre_lens + page_size - 1) // page_size
        num_pages_after = (seq_lens + page_size - 1) // page_size
        num_new_pages = num_pages_after - num_pages_before

        # Calculate new page start location for each request
        new_page_start_locs = torch.zeros(batch_size, dtype=torch.int32, device=pre_lens.device)
        for i in range(1, batch_size):
            new_page_start_locs[i] = new_page_start_locs[i-1] + num_new_pages[i-1]

        # Process each request
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            pre_len = pre_lens[i].item()
            extend_len = seq_len - pre_len
            output_start = output_start_locs[i].item()
            new_page_start = new_page_start_locs[i].item()
            last_loc_val = last_loc[i].item()

            # Part 1: fill the old partial page
            num_part1 = min(seq_len, ((pre_len + page_size - 1) // page_size) * page_size) - pre_len
            if num_part1 > 0:
                out_indices[output_start:output_start + num_part1] = torch.arange(
                    last_loc_val + 1, last_loc_val + 1 + num_part1, dtype=torch.int32, device=pre_lens.device
                )

            if pre_len + num_part1 == seq_len:
                continue

            # Part 2: fill the new full pages
            num_part2 = (seq_len // page_size) * page_size - ((pre_len + page_size - 1) // page_size) * page_size
            if num_part2 > 0:
                indices = []
                for j in range(num_part2):
                    page_idx = new_page_start + j // page_size
                    page_start = free_page[page_idx].item()
                    indices.append(page_start * page_size + j % page_size)
                out_indices[output_start + num_part1:output_start + num_part1 + num_part2] = torch.tensor(
                    indices, dtype=torch.int32, device=pre_lens.device
                )

            if pre_len + num_part1 + num_part2 == seq_len:
                continue

            # Part 3: fill the new partial page
            num_part3 = seq_len - (seq_len // page_size) * page_size
            if num_part3 > 0:
                num_page_self = num_pages_after[i].item() - num_pages_before[i].item()
                page_idx = new_page_start + num_page_self - 1
                page_start = free_page[page_idx].item()
                out_indices[output_start + num_part1 + num_part2:output_start + num_part1 + num_part2 + num_part3] = torch.arange(
                    page_start * page_size, page_start * page_size + num_part3, dtype=torch.int32, device=pre_lens.device
                )

        return out_indices


def get_inputs():
    # Example dimensions
    batch_size = 4
    page_size = 16
    max_seq_len = 128
    dtype = torch.int32

    # Generate test inputs (不使用 device='cuda')
    pre_lens = torch.randint(10, 50, (batch_size,), dtype=dtype)
    seq_lens = pre_lens + torch.randint(5, 30, (batch_size,), dtype=dtype)

    # last_loc should point to the last token position in the last page
    last_loc = torch.empty(batch_size, dtype=dtype)
    for i in range(batch_size):
        pre_len = pre_lens[i].item()
        last_page_idx = (pre_len - 1) // page_size
        last_loc[i] = last_page_idx * page_size + (pre_len - 1) % page_size

    # Calculate total new pages needed
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_pages_after = (seq_lens + page_size - 1) // page_size
    total_new_pages = (num_pages_after - num_pages_before).sum().item()

    # Generate free page indices
    free_page = torch.arange(100, 100 + total_new_pages, dtype=dtype)

    return [pre_lens, seq_lens, last_loc, free_page]


def get_init_inputs():
    return []
