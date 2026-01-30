import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：sglang/python/sglang/srt/layers/attention/fla/l2norm.py
# SGLang函数：l2norm_fwd_kernel1
# 实现类型：Triton kernel
# 功能：L2归一化（小特征维度）
# 测试文件：sglang/test/srt/cpu/test_norm.py
# 输入参考：根据源文件中的函数签名和test_norm.py中的使用方式推断
# ============================================================================

# ============================================================================
# 以下是从SGLang直接复制的Triton Kernel实现
# ============================================================================

@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


def l2norm_fwd_kernel1_impl(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    l2norm_fwd_kernel1[(T,)](
        x,
        y,
        eps=eps,
        D=D,
        BD=BD,
        num_warps=8,
        num_stages=3,
    )

    return y.view(x_shape_og)


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """直接使用复制的Triton Kernel实现"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm_fwd_kernel1_impl(x, eps=self.eps)


class ModelSGLang(nn.Module):
    """sglang实现"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch implementation of L2 normalization
        x_shape_og = x.shape
        x = x.view(-1, x.shape[-1])
        # Compute L2 norm for each row
        norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)) + self.eps
        # Normalize
        y = x / norm
        return y.view(x_shape_og)


def get_inputs():
    """生成测试输入"""
    T = 1024
    D = 256  # Small dimension for l2norm_fwd_kernel1
    dtype = torch.float16

    x = torch.randn(T, D, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return [1e-6]  # eps