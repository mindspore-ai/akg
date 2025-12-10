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
#   generate_draft_decode_kv_indices_kernel - 生成草稿解码的KV索引
# ============================================================================


@triton.jit
def generate_draft_decode_kv_indices_kernel(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid, other=0)
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        extend_data = tl.load(
            token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        prefix_len = seq_len
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_pool_indices, req_to_token, paged_kernel_lens, positions, num_steps, topk, page_size):
        batch_size = paged_kernel_lens.shape[0]
        pool_len = req_to_token.shape[1]

        # Calculate output sizes
        # kv_indices: (num_steps, total_kv_len)
        # For simplicity, we'll calculate max possible size
        max_seq_len = paged_kernel_lens.max().item()
        total_kv_len = (paged_kernel_lens.sum().item() + num_steps) * topk * batch_size

        # kv_indptr: (num_steps, bs * topk + 1)
        kv_indptr_size = batch_size * topk + 1

        # Allocate output tensors
        kv_indices = torch.empty((num_steps, total_kv_len), dtype=req_to_token.dtype, device=req_to_token.device)
        kv_indptr = torch.empty((num_steps, kv_indptr_size), dtype=torch.int32, device=req_to_token.device)

        # Calculate strides
        kv_indices_stride = kv_indices.stride(0)
        kv_indptr_stride = kv_indptr.stride(0)

        # Find next power of 2
        bs_upper = triton.next_power_of_2(batch_size)
        iter_upper = triton.next_power_of_2(num_steps)
        num_tokens_upper = triton.next_power_of_2(batch_size * topk)

        # Launch kernel with 3D grid
        grid = (num_steps, batch_size, topk)
        generate_draft_decode_kv_indices_kernel[grid](
            req_pool_indices,
            req_to_token,
            paged_kernel_lens,
            kv_indices,
            kv_indptr,
            positions,
            pool_len,
            kv_indices_stride,
            kv_indptr_stride,
            bs_upper,
            iter_upper,
            num_tokens_upper,
            page_size,
        )

        return kv_indices, kv_indptr


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, req_pool_indices, req_to_token, paged_kernel_lens, positions, num_steps, topk, page_size):
        batch_size = paged_kernel_lens.shape[0]
        pool_len = req_to_token.shape[1]

        # Calculate output sizes
        max_seq_len = paged_kernel_lens.max().item()
        total_kv_len = (paged_kernel_lens.sum().item() + num_steps) * topk * batch_size

        kv_indptr_size = batch_size * topk + 1

        # Allocate output tensors
        kv_indices = torch.empty((num_steps, total_kv_len), dtype=req_to_token.dtype, device=req_to_token.device)
        kv_indptr = torch.empty((num_steps, kv_indptr_size), dtype=torch.int32, device=req_to_token.device)

        # Process each (step, batch_id, topk_id) combination
        for step in range(num_steps):
            kv_idx = 0
            for bid in range(batch_size):
                seq_len = paged_kernel_lens[bid].item()
                req_pool_idx = req_pool_indices[bid].item()
                token_pool = req_to_token[req_pool_idx]

                for topk_id in range(topk):
                    # Copy prefix tokens
                    kv_indices[step, kv_idx:kv_idx + seq_len] = token_pool[:seq_len]
                    kv_idx += seq_len

                    # Copy extend tokens
                    iters = step + 1
                    if page_size == 1 or topk == 1:
                        start = seq_len + topk_id * num_steps
                        kv_indices[step, kv_idx:kv_idx + iters] = token_pool[start:start + iters]
                    else:
                        prefix_len = seq_len
                        last_page_len = prefix_len % page_size
                        num_new_pages_per_topk = (last_page_len + num_steps + page_size - 1) // page_size
                        prefix_base = (seq_len // page_size) * page_size
                        start = prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
                        kv_indices[step, kv_idx:kv_idx + iters] = token_pool[start:start + iters]

                    kv_idx += iters

            # Update kv_indptr
            for bid in range(batch_size):
                for topk_id in range(topk):
                    zid = bid * topk + topk_id
                    if zid == 0:
                        base = positions[:batch_size * topk].sum().item()
                    else:
                        base = positions[:zid].sum().item()
                    kv_indptr[step, zid] = base + zid * (step + 1)

        return kv_indices, kv_indptr


def get_inputs():
    # Example dimensions
    batch_size = 4
    num_reqs = 8
    pool_len = 512
    num_steps = 5
    topk = 2
    page_size = 16
    dtype = torch.int32

    # req_pool_indices: which request pool each batch element belongs to
    req_pool_indices = torch.randint(0, num_reqs, (batch_size,), dtype=dtype)

    # req_to_token: token pool for each request
    req_to_token = torch.randint(0, 10000, (num_reqs, pool_len), dtype=dtype)

    # paged_kernel_lens: sequence length for each batch element
    paged_kernel_lens = torch.randint(100, 200, (batch_size,), dtype=dtype)

    # positions: position information for each batch element
    positions = torch.randint(50, 150, (batch_size * topk,), dtype=dtype)

    return [req_pool_indices, req_to_token, paged_kernel_lens, positions, num_steps, topk, page_size]


def get_init_inputs():
    return []
