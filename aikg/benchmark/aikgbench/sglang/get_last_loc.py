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
#   get_last_loc_triton(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)
# Triton Kernel:
#   get_last_loc_kernel - 获取每个请求前缀的最后一个token位置
# ============================================================================


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_to_token, req_pool_indices_tensor, prefix_lens_tensor):
        BLOCK_SIZE = 256
        num_tokens = prefix_lens_tensor.shape[0]
        result = torch.empty_like(prefix_lens_tensor)
        grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

        get_last_loc_kernel[grid](
            req_to_token,
            req_pool_indices_tensor,
            prefix_lens_tensor,
            result,
            num_tokens,
            req_to_token.stride(0),
            BLOCK_SIZE,
        )
        return result


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, req_to_token, req_pool_indices_tensor, prefix_lens_tensor):
        from sglang.srt.mem_cache.common import get_last_loc_triton
        return get_last_loc_triton(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_inputs():
    # Example dimensions
    batch_size = 16
    max_context_len = 512
    num_tokens = batch_size
    dtype = torch.int64

    req_to_token = torch.randint(0, 1000, (batch_size, max_context_len), dtype=dtype, device='cuda')
    req_pool_indices_tensor = torch.randint(0, batch_size, (num_tokens,), dtype=dtype, device='cuda')
    prefix_lens_tensor = torch.randint(1, max_context_len, (num_tokens,), dtype=dtype, device='cuda')

    return [req_to_token, req_pool_indices_tensor, prefix_lens_tensor]


def get_init_inputs():
    return []
