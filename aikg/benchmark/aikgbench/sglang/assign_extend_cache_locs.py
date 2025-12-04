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
#   assign_extend_cache_locs_func(req_pool_indices, req_to_token, start_offset,
#                                  end_offset, batch_size, draft_token_num, device)
# Triton Kernel:
#   assign_extend_cache_locs_kernel - EAGLE v2推测解码的缓存位置分配
# ============================================================================


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n."""
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


@triton.jit
def assign_extend_cache_locs_kernel(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_pool_indices, req_to_token, start_offset, end_offset, batch_size, draft_token_num):
        device = req_to_token.device
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs_kernel[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )
        return out_cache_loc


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, req_pool_indices, req_to_token, start_offset, end_offset, batch_size, draft_token_num):
        from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
        device = req_to_token.device
        return assign_extend_cache_locs_func(
            req_pool_indices, req_to_token, start_offset, end_offset, batch_size, draft_token_num, device
        )


def get_inputs():
    # Example dimensions for EAGLE v2 speculative decoding
    batch_size = 8
    pool_len = 512
    draft_token_num = 16
    dtype = torch.int64

    req_to_token = torch.randint(0, 10000, (batch_size, pool_len), dtype=dtype, device='cuda')
    req_pool_indices = torch.arange(batch_size, dtype=dtype, device='cuda')

    start_offset = torch.randint(0, pool_len - draft_token_num, (batch_size,), dtype=dtype, device='cuda')
    end_offset = start_offset + torch.randint(1, draft_token_num, (batch_size,), dtype=dtype, device='cuda')

    return [req_pool_indices, req_to_token, start_offset, end_offset, batch_size, draft_token_num]


def get_init_inputs():
    return []
