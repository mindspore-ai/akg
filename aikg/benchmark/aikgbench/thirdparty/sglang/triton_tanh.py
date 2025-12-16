import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/attention/triton_ops/extend_attention.py
# 测试文件: 无 (tanh是辅助函数)
# SGLang实现:
#   @triton.jit def tanh(x): return 2 * tl.sigmoid(2 * x) - 1
# 注意:
#   这是一个Triton辅助函数，被多个attention kernel调用用于logit_cap
# ============================================================================


@triton.jit
def triton_tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _tanh_kernel(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = triton_tanh(x)
    tl.store(Y + offsets, y, mask=mask)


def triton_tanh_impl(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _tanh_kernel[grid](x.view(-1), y.view(-1), n_elements, BLOCK_SIZE)
    return y


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh_impl(x)


class ModelSglang(nn.Module):
    """PyTorch tanh对照"""

    def __init__(self):
        super(ModelSglang, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch torch.tanh作为参考
        """
        return torch.tanh(x)


def get_inputs():
    batch_size = 4
    seq_len = 512
    num_heads = 32
    dtype = torch.float32
    x = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=dtype)
    return [x]


def get_init_inputs():
    return []
