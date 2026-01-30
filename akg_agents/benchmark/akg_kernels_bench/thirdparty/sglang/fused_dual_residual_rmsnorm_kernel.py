import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/elementwise.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps, autotune=False)
# Triton Kernel:
#   fused_dual_residual_rmsnorm_kernel - 融合双残差RMSNorm
# ============================================================================


@triton.jit
def fused_dual_residual_rmsnorm_kernel(
    output_ptr,
    mid_ptr,
    activ_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(mid_ptr + input_start + offsets, a2r, mask=mask)

    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)

    tl.store(output_ptr + input_start + offsets, a2r / rms2 * w2, mask=mask)


def fused_dual_residual_rmsnorm_impl(x, residual, weight1, weight2, eps):
    assert len(x.shape) == 2
    assert x.shape == residual.shape and x.dtype == residual.dtype
    output, mid = torch.empty_like(x), torch.empty_like(x)
    bs, hidden_dim = x.shape
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4),
    }

    fused_dual_residual_rmsnorm_kernel[(bs,)](
        output, mid, x, residual, weight1, weight2, eps=eps, hidden_dim=hidden_dim, **config
    )
    return output, mid


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self, eps=1e-6):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x, residual, weight1, weight2) -> Tuple[torch.Tensor, torch.Tensor]:
        return fused_dual_residual_rmsnorm_impl(x, residual, weight1, weight2, self.eps)


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self, eps=1e-6):
        super(ModelSglang, self).__init__()
        self.eps = eps

    def forward(self, x, residual, weight1, weight2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        调用SGLang fused_dual_residual_rmsnorm API
        """
        from sglang.srt.layers.elementwise import fused_dual_residual_rmsnorm

        return fused_dual_residual_rmsnorm(x, residual, weight1, weight2, self.eps, autotune=False)


def get_inputs():
    batch_size = 32
    hidden_dim = 4096
    dtype = torch.float16
    x = torch.randn(batch_size, hidden_dim, dtype=dtype)
    residual = torch.randn(batch_size, hidden_dim, dtype=dtype)
    weight1 = torch.randn(hidden_dim, dtype=dtype)
    weight2 = torch.randn(hidden_dim, dtype=dtype)
    return [x, residual, weight1, weight2]


def get_init_inputs():
    eps = 1e-6
    return [eps]

