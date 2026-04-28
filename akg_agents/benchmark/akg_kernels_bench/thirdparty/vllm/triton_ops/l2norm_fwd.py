import torch
from typing import Optional
import torch.nn as nn

import triton
import triton.language as tl

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/fla/ops/l2norm.py
# vLLM函数: l2norm_fwd
# 实现类型: Triton kernel
# 功能: L2归一化
# 测试文件: 无（vLLM仓库中未找到专门的l2norm测试文件）
# 输入参考: 根据源文件中的函数签名和chunk.py中的使用方式推断
# ============================================================================

# ============================================================================
# 以下是从vLLM直接复制的Triton kernel实现
# ============================================================================


@triton.jit
def l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr):
    """从vLLM复制的Triton kernel"""
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, N)[None, :]
    xs = tl.load(X + (rindex + N * row_idx), xmask).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, N])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


def l2norm_fwd_impl(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """从vLLM复制的host侧调用代码"""
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])

    # 分配输出
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)

    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]

    # 检查特征维度限制
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    MBLOCK = 32
    l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](
        x,
        y,
        eps,
        T,
        D,
        MBLOCK,
    )

    return y.view(x_shape_og)


# ============================================================================
# AIKGBench标准接口
# ============================================================================


class Model(nn.Module):
    """直接使用复制的Triton kernel实现"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm_fwd_impl(x, eps=self.eps)


class ModelVLLM(nn.Module):
    """通过vLLM库调用"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
        return l2norm_fwd(x, eps=self.eps)


def get_inputs():
    """生成测试输入"""
    T = 1024
    D = 512
    dtype = torch.float16

    x = torch.randn(T, D, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return [1e-6]  # eps

