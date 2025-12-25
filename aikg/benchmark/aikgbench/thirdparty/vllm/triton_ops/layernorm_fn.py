# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/fla/ops/layernorm_guard.py
# vLLM函数: layernorm_fn, rmsnorm_fn
# 实现类型: Triton kernel
# 测试文件: 无专门测试文件
# 输入参考: x=[batch*seq, hidden], weight=[hidden], bias=[hidden](可选)
# ============================================================================

import torch
import torch.nn as nn

# ============================================================================
# 从vLLM复制的Triton kernel实现
# ============================================================================
import triton
import triton.language as tl


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        x *= z * tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        mean = 0.0

    rstd = tl.rsqrt(var + eps)

    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        y *= z * tl.sigmoid(z)

    tl.store(Y_base, y, mask=mask)


def layer_norm_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
):
    """从vLLM复制的host侧调用代码"""
    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size
    
    out = torch.empty_like(x)
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    rows_per_block = min(triton.next_power_of_2(triton.cdiv(M, 128)), 4)
    grid = (triton.cdiv(M, rows_per_block), ngroups)
    
    layer_norm_fwd_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    return out


def layernorm_fn_impl(x, weight, bias, eps=1e-6, is_rms_norm=False):
    """简化的layernorm前向函数"""
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = layer_norm_fwd_impl(x, weight, bias, eps, is_rms_norm=is_rms_norm)
    return y.reshape(x_shape_og)


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """原生实现（直接调用复制的Triton kernel）"""
    
    def __init__(self, hidden_size: int = 1024, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm_fn_impl(x, self.weight, self.bias, eps=self.eps, is_rms_norm=False)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""
    
    def __init__(self, hidden_size: int = 1024, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import layernorm_fn
        return layernorm_fn(x, self.weight, self.bias, eps=self.eps)


def get_inputs():
    """生成测试输入"""
    batch_size = 4
    seq_len = 128
    hidden_size = 1024
    
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    
    return [x]


def get_init_inputs():
    """获取初始化参数"""
    return [1024, 1e-6]  # hidden_size, eps

