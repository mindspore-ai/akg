import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/attention/triton_ops/merge_state.py
# 测试文件: sgl-kernel/tests/test_merge_state.py, test_merge_state_v2.py
# SGLang API调用:
#   merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse)
# Triton Kernel:
#   merge_state_kernel - Merges two attention states (prefix and suffix)
# 标杆实现:
#   sgl_kernel.merge_state_v2 - CUDA implementation
# ============================================================================


@triton.jit
def merge_state_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_merged
    output_lse,  # [NUM_TOKENS, NUM_HEADS] s_merged
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_a
    prefix_lse,  # [NUM_TOKENS, NUM_HEADS] s_a
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_b
    suffix_lse,  # [NUM_TOKENS, NUM_HEADS] s_b
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + token_idx * num_heads + head_idx)
    s_lse = tl.load(suffix_lse + token_idx * num_heads + head_idx)
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    out_se = tl.exp(p_lse) + tl.exp(s_lse)

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + token_idx * num_heads + head_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )

    p_scale = tl.exp(p_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out,
        mask=head_mask,
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prefix_output: [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] - prefix attention output
            prefix_lse: [NUM_TOKENS, NUM_HEADS] - prefix log-sum-exp values
            suffix_output: [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] - suffix attention output
            suffix_lse: [NUM_TOKENS, NUM_HEADS] - suffix log-sum-exp values
        Returns:
            output: [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] - merged attention output
            output_lse: [NUM_TOKENS, NUM_HEADS] - merged log-sum-exp values
        """
        output = torch.empty_like(prefix_output)
        output_lse = torch.empty_like(prefix_lse)

        num_tokens = output.shape[0]
        num_query_heads = output.shape[1]
        head_size = output.shape[2]
        padded_head_size = triton.next_power_of_2(head_size)

        merge_state_kernel[(num_tokens, num_query_heads)](
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            head_size,
            padded_head_size,
            output_lse is not None,
        )
        return output, output_lse


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(
        self,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reference implementation using sgl_kernel CUDA implementation
        """
        from sgl_kernel import merge_state_v2

        output = torch.empty_like(prefix_output)
        output_lse = torch.empty_like(prefix_lse)

        return merge_state_v2(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )


def get_inputs():
    # Example dimensions for merge state
    num_tokens = 128
    num_heads = 32
    head_size = 128
    dtype = torch.float16

    prefix_output = torch.randn(
        (num_tokens, num_heads, head_size),
        dtype=dtype,
        device='cuda'
    )
    prefix_lse = torch.randn(
        (num_tokens, num_heads),
        dtype=torch.float32,
        device='cuda'
    )
    suffix_output = torch.randn(
        (num_tokens, num_heads, head_size),
        dtype=dtype,
        device='cuda'
    )
    suffix_lse = torch.randn(
        (num_tokens, num_heads),
        dtype=torch.float32,
        device='cuda'
    )

    return [prefix_output, prefix_lse, suffix_output, suffix_lse]


def get_init_inputs():
    return []
