import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/mem_cache/common.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   write_req_to_token_pool_triton(req_to_token, req_pool_indices, prefix_tensors,
#                                   prefix_lens, seq_lens, extend_lens,
#                                   out_cache_loc, req_to_token.shape[1])
# Triton Kernel:
#   write_req_to_token_pool_triton_kernel - 将请求的前缀和扩展token写入token池
# ============================================================================


@triton.jit
def write_req_to_token_pool_triton_kernel(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_to_token, req_pool_indices, prefix_tensors_list, prefix_lens,
                seq_lens, extend_lens, out_cache_loc):
        # Convert prefix tensors list to pointer array
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors_list],
            device=req_to_token.device,
            dtype=torch.uint64,
        )

        write_req_to_token_pool_triton_kernel[(req_pool_indices.shape[0],)](
            req_to_token,
            req_pool_indices,
            prefix_pointers,
            prefix_lens,
            seq_lens,
            extend_lens,
            out_cache_loc,
            req_to_token.shape[1],
        )

        return req_to_token


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, req_to_token, req_pool_indices, prefix_tensors_list, prefix_lens,
                seq_lens, extend_lens, out_cache_loc):
        from sglang.srt.mem_cache.common import write_req_to_token_pool_triton

        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors_list],
            device=req_to_token.device,
            dtype=torch.uint64,
        )

        write_req_to_token_pool_triton[(req_pool_indices.shape[0],)](
            req_to_token,
            req_pool_indices,
            prefix_pointers,
            prefix_lens,
            seq_lens,
            extend_lens,
            out_cache_loc,
            req_to_token.shape[1],
        )

        return req_to_token


def get_inputs():
    # Example dimensions
    batch_size = 8
    max_context_len = 512
    max_prefix_len = 128
    max_extend_len = 64
    dtype = torch.int64

    req_to_token = torch.zeros(batch_size, max_context_len, dtype=dtype, device='cuda')
    req_pool_indices = torch.arange(batch_size, dtype=dtype, device='cuda')

    # Create prefix tensors for each request
    prefix_tensors_list = []
    prefix_lens_list = []
    seq_lens_list = []
    extend_lens_list = []

    total_extend_tokens = 0
    for i in range(batch_size):
        prefix_len = torch.randint(10, max_prefix_len, (1,)).item()
        extend_len = torch.randint(10, max_extend_len, (1,)).item()
        seq_len = prefix_len + extend_len

        prefix_tensor = torch.randint(0, 1000, (prefix_len,), dtype=dtype, device='cuda')
        prefix_tensors_list.append(prefix_tensor)
        prefix_lens_list.append(prefix_len)
        seq_lens_list.append(seq_len)
        extend_lens_list.append(extend_len)
        total_extend_tokens += extend_len

    prefix_lens = torch.tensor(prefix_lens_list, dtype=dtype, device='cuda')
    seq_lens = torch.tensor(seq_lens_list, dtype=dtype, device='cuda')
    extend_lens = torch.tensor(extend_lens_list, dtype=dtype, device='cuda')
    out_cache_loc = torch.randint(0, 10000, (total_extend_tokens,), dtype=dtype, device='cuda')

    return [req_to_token, req_pool_indices, prefix_tensors_list, prefix_lens,
            seq_lens, extend_lens, out_cache_loc]


def get_init_inputs():
    return []
