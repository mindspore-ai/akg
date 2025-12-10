import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/elementwise.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   experts_combine_triton(moe_hidden_states, mlp_hidden_states, output_buffer=None)
# Triton Kernel:
#   experts_combine_kernel - MoE专家合并
# ============================================================================


@triton.jit
def experts_combine_kernel(
    out_hidden_states,
    moe_hidden_states,
    mlp_hidden_states,
    combine_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index_mlp = pid * hidden_dim
    start_index_rmoe = pid * hidden_dim * combine_k
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    combine_k_offsets = tl.arange(0, combine_k)

    moe_x = tl.load(
        moe_hidden_states + start_index_rmoe + combine_k_offsets[:, None] * hidden_dim + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x = tl.sum(moe_x, axis=0)
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)
    combined_x = (moe_x + mlp_x) / 1.4142135623730951

    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)


def experts_combine_impl(
    moe_hidden_states: torch.Tensor,
    mlp_hidden_states: torch.Tensor,
    output_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert moe_hidden_states.is_contiguous()
    assert mlp_hidden_states.is_contiguous()

    if len(moe_hidden_states.shape) == 2:
        combine_k = 1
    else:
        combine_k = moe_hidden_states.shape[1]

    if output_buffer is None:
        out_hidden_states = torch.empty_like(mlp_hidden_states)
    else:
        flat_output_buffer = output_buffer.view(mlp_hidden_states.dtype).reshape(-1)
        assert flat_output_buffer.numel() >= mlp_hidden_states.numel()
        out_hidden_states = flat_output_buffer[: mlp_hidden_states.numel()].reshape(mlp_hidden_states.shape)

    bs, hidden_dim = mlp_hidden_states.shape

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4),
    }

    experts_combine_kernel[(bs,)](
        out_hidden_states, moe_hidden_states, mlp_hidden_states, combine_k, hidden_dim, **config
    )

    return out_hidden_states


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, moe_hidden_states, mlp_hidden_states):
        return experts_combine_impl(moe_hidden_states, mlp_hidden_states)


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self):
        super(ModelSglang, self).__init__()

    def forward(self, moe_hidden_states, mlp_hidden_states):
        """
        调用SGLang experts_combine_triton API
        """
        from sglang.srt.layers.elementwise import experts_combine_triton

        return experts_combine_triton(moe_hidden_states, mlp_hidden_states)


def get_inputs():
    batch_size = 32
    hidden_dim = 4096
    combine_k = 4
    dtype = torch.float16
    moe_hidden_states = torch.randn(batch_size, combine_k, hidden_dim, dtype=dtype)
    mlp_hidden_states = torch.randn(batch_size, hidden_dim, dtype=dtype)
    return [moe_hidden_states, mlp_hidden_states]


def get_init_inputs():
    return []

