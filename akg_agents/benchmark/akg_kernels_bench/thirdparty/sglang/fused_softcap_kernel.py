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
#   fused_softcap(x, softcap_const, autotune=False)
# Triton Kernel:
#   fused_softcap_kernel - softcap操作 (tanh(x/softcap) * softcap)
# ============================================================================


@triton.jit
def fused_softcap_kernel(
    output_ptr,
    input_ptr,
    n_ele,
    softcap_const: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele
    x = tl.load(input_ptr + offsets, mask=mask)
    fx = x.to(tl.float32)
    fxs = fx / softcap_const
    exped = tl.exp(2 * fxs)
    top = exped - 1
    bottom = exped + 1
    output = top / bottom * softcap_const
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_softcap_impl(x, softcap_const):
    output = torch.empty_like(x, dtype=torch.float32)
    n_elements = output.numel()
    fused_softcap_kernel[(triton.cdiv(n_elements, 128),)](
        output, x, n_elements, softcap_const, BLOCK_SIZE=128, num_warps=8
    )
    return output


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self, softcap_const):
        super(Model, self).__init__()
        self.softcap_const = softcap_const

    def forward(self, x):
        return fused_softcap_impl(x, self.softcap_const)


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self, softcap_const):
        super(ModelSglang, self).__init__()
        self.softcap_const = softcap_const

    def forward(self, x):
        """
        调用SGLang fused_softcap API
        """
        from sglang.srt.layers.elementwise import fused_softcap

        return fused_softcap(x, self.softcap_const, autotune=False)


def get_inputs():
    batch_size = 32
    seq_len = 512
    hidden_dim = 4096
    dtype = torch.float16
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype)
    return [x]


def get_init_inputs():
    softcap_const = 30.0
    return [softcap_const]

