import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/elementwise.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   fused_rmsnorm(x, weight, eps, autotune=False, inplace=False)
# Triton Kernel:
#   fused_rmsnorm_kernel - 融合RMSNorm
# ============================================================================


@triton.jit
def fused_rmsnorm_kernel(
    output_ptr,
    activ_ptr,
    weight_ptr,
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

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a_rms = a / rms * w1

    tl.store(output_ptr + input_start + offsets, a_rms, mask=mask)


def fused_rmsnorm_impl(x, weight, eps, inplace=False):
    assert len(x.shape) == 2
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    bs, hidden_dim = x.shape
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), 32), 4),
    }

    fused_rmsnorm_kernel[(bs,)](output, x, weight, eps=eps, hidden_dim=hidden_dim, **config)
    return output


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self, eps=1e-6):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x, weight):
        return fused_rmsnorm_impl(x, weight, self.eps)


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self, eps=1e-6):
        super(ModelSglang, self).__init__()
        self.eps = eps

    def forward(self, x, weight):
        """
        调用SGLang fused_rmsnorm API
        """
        from sglang.srt.layers.elementwise import fused_rmsnorm

        return fused_rmsnorm(x, weight, self.eps, autotune=False, inplace=False)


def get_inputs():
    batch_size = 32
    hidden_dim = 4096
    dtype = torch.float16
    x = torch.randn(batch_size, hidden_dim, dtype=dtype)
    weight = torch.randn(hidden_dim, dtype=dtype)
    return [x, weight]


def get_init_inputs():
    eps = 1e-6
    return [eps]

