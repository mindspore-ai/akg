import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：sglang/benchmark/kernels/scheduler_batch/benchmark_get_last_loc_triton.py
# SGLang函数：get_last_loc_triton
# 实现类型：Triton kernel
# 功能：获取请求池索引对应的最后位置标记
# 测试文件：sglang/benchmark/kernels/scheduler_batch/benchmark_get_last_loc_triton.py
# 输入参考：根据源文件中的函数签名和使用方式推断
# ============================================================================

# ============================================================================
# 以下是从SGLang直接复制的Triton Kernel实现
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



def get_last_loc_triton_impl(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
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

# ============================================================================
# AIKGBench标准接口
# ============================================================================
class Model(nn.Module):
    """直接使用复制的Triton Kernel实现"""
    def __init__(self):
        super().__init__()
        # 不需要额外的初始化参数
    
    def forward(self, req_to_token: torch.Tensor, req_pool_indices_tensor: torch.Tensor, prefix_lens_tensor: torch.Tensor) -> torch.Tensor:
        return get_last_loc_triton_impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)

class ModelSGLang(nn.Module):
    """PyTorch实现"""

    def __init__(self):
        super().__init__()
        # 不需要额外的初始化参数
    
    def forward(self, req_to_token: torch.Tensor, req_pool_indices_tensor: torch.Tensor, prefix_lens_tensor: torch.Tensor) -> torch.Tensor:
        return torch.where(
            prefix_lens_tensor > 0,
            req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
            torch.full_like(prefix_lens_tensor, -1),
        )

def get_inputs():
    """生成测试输入"""
    max_batch = 4097
    max_context_len = 6148
    batch_size = 20
    dtype_req_to_token = torch.int32
    dtype_indices = torch.int64
    
    # 生成req_to_token
    req_to_token = torch.zeros(
        (max_batch, max_context_len), dtype=dtype_req_to_token
    )
    
    # 生成req_pool_indices
    req_pool_indices = torch.arange(batch_size, dtype=dtype_indices)
    
    # 生成prefix_lens
    prefix_lens = torch.randint(
        -max_context_len // 2,
        max_context_len,
        (batch_size,),
        dtype=dtype_indices
    )
    
    return [req_to_token, req_pool_indices, prefix_lens]

def get_init_inputs():
    """生成初始化参数"""
    # 该模型不需要初始化参数
    return []