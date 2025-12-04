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
#   assign_draft_cache_locs_kernel - 为草稿token分配缓存位置
# ============================================================================


@triton.jit
def assign_draft_cache_locs_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len: tl.constexpr,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(tl.load(extend_lens + bs_offset, mask=bs_offset < pid))
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)

    if page_size != 1 and topk != 1 and duplicate_cache_len > 0:
        # Part 2: Copy indices into source_cache_loc and target_cache_loc
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        offsets = tl.arange(0, page_size)
        mask = offsets < last_page_len
        num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
        prefix_base = token_pool + prefix_len - last_page_len
        src_indices = tl.load(prefix_base + offsets, mask=mask)
        last_page_lens_cumsum_ = tl.load(last_page_lens_cumsum + pid)

        # Skip the first one since no copy is needed
        for topk_id in range(1, topk):
            tl.store(
                source_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                src_indices,
                mask=mask,
            )
            tgt_indices = tl.load(
                prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
                mask=mask,
            )
            tl.store(
                target_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                tgt_indices,
                mask=mask,
            )

        # Part 3: Copy and remove the used indices for duplication
        iter_offset = tl.arange(0, iter_upper)
        for topk_id in range(topk):
            mask_upper = iter_offset < (speculative_num_steps + last_page_len)
            mask_lower = iter_offset >= last_page_len
            combined_mask = mask_upper & mask_lower
            indices = tl.load(
                prefix_base
                + topk_id * num_new_pages_per_topk_ * page_size
                + iter_offset,
                mask=combined_mask,
                other=0,
            )
            # Shift from previous batches
            ptr_offset = pid * speculative_num_steps * topk
            # Subtract last_page_len to fill the gap of duplicated last page tokens.
            tl.store(
                out_cache_loc
                + ptr_offset
                + topk_id * speculative_num_steps
                - last_page_len
                + iter_offset,
                indices,
                mask=combined_mask,
            )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_pool_indices, req_to_token, seq_lens, extend_lens, num_new_pages_per_topk,
                out_cache_loc, last_page_lens_cumsum, duplicate_cache_len, topk, speculative_num_steps, page_size):
        batch_size = seq_lens.shape[0]
        pool_len = req_to_token.shape[1]

        # Allocate source_cache_loc and target_cache_loc if needed
        if page_size != 1 and topk != 1 and duplicate_cache_len > 0:
            # Calculate size based on last_page_lens
            last_page_lens = seq_lens % page_size
            source_size = (topk - 1) * last_page_lens.sum().item()
            source_cache_loc = torch.empty(source_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
            target_cache_loc = torch.empty(source_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
        else:
            source_cache_loc = torch.empty(0, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
            target_cache_loc = torch.empty(0, dtype=out_cache_loc.dtype, device=out_cache_loc.device)

        # Find next power of 2
        bs_upper = triton.next_power_of_2(batch_size)
        iter_upper = triton.next_power_of_2(max(speculative_num_steps + page_size, 1))

        # Launch kernel
        grid = (batch_size,)
        assign_draft_cache_locs_kernel[grid](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            pool_len,
            topk,
            speculative_num_steps,
            page_size,
            bs_upper,
            iter_upper,
        )

        return req_to_token, out_cache_loc, source_cache_loc, target_cache_loc


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, req_pool_indices, req_to_token, seq_lens, extend_lens, num_new_pages_per_topk,
                out_cache_loc, last_page_lens_cumsum, duplicate_cache_len, topk, speculative_num_steps, page_size):
        batch_size = seq_lens.shape[0]
        pool_len = req_to_token.shape[1]

        # Clone inputs to avoid modifying them
        req_to_token = req_to_token.clone()
        out_cache_loc = out_cache_loc.clone()

        # Allocate source_cache_loc and target_cache_loc if needed
        if page_size != 1 and topk != 1 and duplicate_cache_len > 0:
            last_page_lens = seq_lens % page_size
            source_size = (topk - 1) * last_page_lens.sum().item()
            source_cache_loc = torch.empty(source_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
            target_cache_loc = torch.empty(source_size, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
        else:
            source_cache_loc = torch.empty(0, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
            target_cache_loc = torch.empty(0, dtype=out_cache_loc.dtype, device=out_cache_loc.device)

        # Process each batch element
        for pid in range(batch_size):
            # Determine copy length and out_cache_ptr
            if page_size == 1 or topk == 1:
                copy_len = topk * speculative_num_steps
                out_cache_start = pid * topk * speculative_num_steps
            else:
                copy_len = extend_lens[pid].item()
                out_cache_start = extend_lens[:pid].sum().item()

            # Part 1: Copy from out_cache_loc to req_to_token
            kv_start = seq_lens[pid].item()
            req_pool_idx = req_pool_indices[pid].item()
            req_to_token[req_pool_idx, kv_start:kv_start + copy_len] = out_cache_loc[out_cache_start:out_cache_start + copy_len]

            if page_size != 1 and topk != 1 and duplicate_cache_len > 0:
                # Part 2: Copy indices into source_cache_loc and target_cache_loc
                prefix_len = seq_lens[pid].item()
                last_page_len = prefix_len % page_size
                num_new_pages_per_topk_ = num_new_pages_per_topk[pid].item()
                prefix_base_idx = prefix_len - last_page_len
                last_page_lens_cumsum_ = last_page_lens_cumsum[pid].item()

                if last_page_len > 0:
                    src_indices = req_to_token[req_pool_idx, prefix_base_idx:prefix_base_idx + last_page_len]

                    for topk_id in range(1, topk):
                        src_start = (topk - 1) * (last_page_lens_cumsum_ - last_page_len) + (topk_id - 1) * last_page_len
                        source_cache_loc[src_start:src_start + last_page_len] = src_indices

                        tgt_start_in_pool = prefix_base_idx + topk_id * num_new_pages_per_topk_ * page_size
                        tgt_indices = req_to_token[req_pool_idx, tgt_start_in_pool:tgt_start_in_pool + last_page_len]
                        target_cache_loc[src_start:src_start + last_page_len] = tgt_indices

                # Part 3: Copy and remove the used indices for duplication
                for topk_id in range(topk):
                    # Read range: [last_page_len, speculative_num_steps + last_page_len)
                    read_start = prefix_base_idx + topk_id * num_new_pages_per_topk_ * page_size + last_page_len
                    read_len = speculative_num_steps
                    indices = req_to_token[req_pool_idx, read_start:read_start + read_len]

                    # Write to out_cache_loc
                    write_start = pid * speculative_num_steps * topk + topk_id * speculative_num_steps - last_page_len
                    out_cache_loc[write_start:write_start + read_len] = indices

        return req_to_token, out_cache_loc, source_cache_loc, target_cache_loc


def get_inputs():
    # Example dimensions
    batch_size = 4
    num_reqs = 8
    pool_len = 512
    topk = 3
    speculative_num_steps = 5
    page_size = 16
    dtype = torch.int32

    # req_pool_indices: which request pool each batch element belongs to
    req_pool_indices = torch.randint(0, num_reqs, (batch_size,), dtype=dtype)

    # req_to_token: token pool for each request
    req_to_token = torch.randint(0, 10000, (num_reqs, pool_len), dtype=dtype)

    # seq_lens: current sequence length for each batch element
    seq_lens = torch.randint(100, 200, (batch_size,), dtype=dtype)

    # extend_lens: extension length for each batch element
    if page_size == 1 or topk == 1:
        extend_lens = torch.full((batch_size,), topk * speculative_num_steps, dtype=dtype)
    else:
        # Calculate based on page alignment
        last_page_lens = seq_lens % page_size
        num_new_pages_per_topk_val = ((last_page_lens + speculative_num_steps + page_size - 1) // page_size)
        extend_lens = num_new_pages_per_topk_val * page_size * topk

    # num_new_pages_per_topk: number of new pages per topk for each batch element
    num_new_pages_per_topk = ((seq_lens % page_size + speculative_num_steps + page_size - 1) // page_size)

    # out_cache_loc: output cache locations
    if page_size == 1 or topk == 1:
        out_cache_loc_size = batch_size * topk * speculative_num_steps
    else:
        out_cache_loc_size = extend_lens.sum().item()
    out_cache_loc = torch.randint(0, 10000, (out_cache_loc_size,), dtype=dtype)

    # last_page_lens_cumsum: cumulative sum of last page lengths
    last_page_lens = seq_lens % page_size
    last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)

    # duplicate_cache_len: whether to enable duplicate cache handling
    duplicate_cache_len = 1 if (page_size != 1 and topk != 1) else 0

    return [
        req_pool_indices, req_to_token, seq_lens, extend_lens, num_new_pages_per_topk,
        out_cache_loc, last_page_lens_cumsum, duplicate_cache_len, topk, speculative_num_steps, page_size
    ]


def get_init_inputs():
    return []
