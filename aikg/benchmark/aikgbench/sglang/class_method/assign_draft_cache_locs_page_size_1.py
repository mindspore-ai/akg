import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/speculative/eagle_info_v2.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   内部使用,用于EAGLE v2推测解码(page_size=1)
# Triton Kernel:
#   assign_draft_cache_locs_page_size_1_kernel - 为草稿token分配缓存位置(page_size=1)
# ============================================================================


@triton.jit
def assign_draft_cache_locs_page_size_1_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_pool_indices, req_to_token, seq_lens, topk, speculative_num_steps):
        bs = len(seq_lens)
        out_cache_loc = torch.empty(
            (bs * topk * speculative_num_steps,),
            dtype=torch.int64,
            device=req_pool_indices.device,
        )
        pool_len = req_to_token.shape[1]

        assign_draft_cache_locs_page_size_1_kernel[(bs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            pool_len,
            topk,
            speculative_num_steps,
        )

        return out_cache_loc


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, req_pool_indices, req_to_token, seq_lens, topk, speculative_num_steps):
        bs = len(seq_lens)
        copy_len = topk * speculative_num_steps
        out_cache_loc = torch.empty(
            (bs * copy_len,),
            dtype=torch.int64,
            device=req_pool_indices.device,
        )

        # PyTorch implementation
        for pid in range(bs):
            kv_start = seq_lens[pid].item()
            pool_idx = req_pool_indices[pid].item()
            token_pool = req_to_token[pool_idx]

            # Copy copy_len elements from token_pool starting at kv_start
            out_start = pid * copy_len
            out_cache_loc[out_start:out_start + copy_len] = token_pool[kv_start:kv_start + copy_len]

        return out_cache_loc


def get_inputs():
    # Example dimensions for Eagle V2 speculative decoding
    batch_size = 8
    pool_len = 2048  # max sequence length per request
    topk = 4  # top-k for draft tokens
    speculative_num_steps = 5  # number of speculative steps
    dtype = torch.int64

    # Create test inputs (not using device='cuda' as requested)
    req_pool_indices = torch.randint(0, batch_size, (batch_size,), dtype=dtype)
    req_to_token = torch.randint(0, 50000, (batch_size, pool_len), dtype=dtype)
    seq_lens = torch.randint(100, pool_len - topk * speculative_num_steps, (batch_size,), dtype=dtype)

    return [req_pool_indices, req_to_token, seq_lens, topk, speculative_num_steps]


def get_init_inputs():
    return []
