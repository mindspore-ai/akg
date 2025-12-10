import torch
from typing import Optional
import torch.nn as nn

import triton
import triton.language as tl

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/attention/ops/triton_merge_attn_states.py
# vLLM函数: merge_attn_states
# 实现类型: Triton kernel
# 功能: 合并注意力状态（用于split-KV场景）
# 测试文件: tests/kernels/attention/test_merge_attn_states.py
# 输入参考:
#   - NUM_BATCH_TOKENS = [256, 512, 613, 1024, 1536, 4096]
#   - NUM_QUERY_HEADS = [4, 8, 16, 32, 48, 64]
#   - HEAD_SIZES = [32, 48, 64, 96, 128, 256]
#   - DTYPES = [torch.float32, torch.half, torch.bfloat16]
# ============================================================================

# ============================================================================
# 以下是从vLLM直接复制的Triton kernel实现
# 源文件: vllm/attention/ops/triton_merge_attn_states.py
# ============================================================================


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    """从vLLM复制的Triton kernel"""
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 and FA3 have different behavior for when the sum-exp is 0
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )


def merge_attn_states_impl(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    """从vLLM复制的host侧调用代码"""
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)

    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        head_size,
        padded_head_size,
        output_lse is not None,
    )


# ============================================================================
# AIKGBench标准接口
# ============================================================================


class Model(nn.Module):
    """直接使用复制的Triton kernel实现"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(prefix_output)
        merge_attn_states_impl(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse
        )
        return output


class ModelVLLM(nn.Module):
    """通过vLLM库调用"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.attention.ops.triton_merge_attn_states import merge_attn_states

        output = torch.empty_like(prefix_output)
        merge_attn_states(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse
        )
        return output


def get_inputs():
    """
    生成测试输入
    参考: tests/kernels/attention/test_merge_attn_states.py
    NUM_BATCH_TOKENS = [256, 512, 613, 1024, 1536, 4096]
    NUM_QUERY_HEADS = [4, 8, 16, 32, 48, 64]
    HEAD_SIZES = [32, 48, 64, 96, 128, 256]
    """
    num_tokens = 512
    num_heads = 16
    head_size = 64
    dtype = torch.float16

    prefix_output = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    prefix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32)
    suffix_output = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    suffix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32)

    return [prefix_output, prefix_lse, suffix_output, suffix_lse]


def get_init_inputs():
    """生成初始化参数"""
    return []

