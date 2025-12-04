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
#   assign_req_to_token_pool_func(req_pool_indices, req_to_token, start_offset,
#                                  end_offset, out_cache_loc, batch_size)
# Triton Kernel:
#   assign_req_to_token_pool_kernel - 将请求分配到token池的推测解码工具
# ============================================================================


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n."""
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


@triton.jit
def assign_req_to_token_pool_kernel(
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

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, batch_size):
        assign_req_to_token_pool_kernel[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )
        return req_to_token


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, batch_size):
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
        assign_req_to_token_pool_func(
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, batch_size
        )
        return req_to_token


def get_inputs():
    # Example dimensions for speculative decoding
    batch_size = 8
    pool_len = 512
    max_tokens_per_req = 64
    dtype = torch.int64

    req_to_token = torch.zeros(batch_size, pool_len, dtype=dtype, device='cuda')
    req_pool_indices = torch.arange(batch_size, dtype=dtype, device='cuda')

    start_offset = torch.randint(0, 100, (batch_size,), dtype=dtype, device='cuda')
    end_offset = start_offset + torch.randint(10, max_tokens_per_req, (batch_size,), dtype=dtype, device='cuda')

    total_tokens = (end_offset - start_offset).sum().item()
    out_cache_loc = torch.randint(0, 10000, (total_tokens,), dtype=dtype, device='cuda')

    return [req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, batch_size]


def get_init_inputs():
    return []
