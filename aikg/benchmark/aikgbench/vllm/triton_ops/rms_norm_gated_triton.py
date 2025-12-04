# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/mamba/ops/layernorm_gated.py
# vLLM函数: rms_norm_gated
# 实现类型: Triton kernel
# 测试文件: 无专门测试文件
# 输入参考: x=[batch*seq, hidden], weight=[hidden], z=[batch*seq, hidden] (可选)
# ============================================================================

import torch
import torch.nn as nn

# ============================================================================
# 从vLLM复制的Triton kernel实现
# ============================================================================
import triton
import triton.language as tl


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row: tl.int64,
    stride_y_row: tl.int64,
    stride_z_row: tl.int64,
    M: tl.int64,
    N: tl.int64,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd_impl(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    """从vLLM复制的host侧调用代码"""
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    
    if out is not None:
        assert out.shape == x.shape
    else:
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
    grid = (M, ngroups)
    
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](
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
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
        )
    return out, mean, rstd


def rms_norm_gated_impl(
    x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True
):
    """从vLLM复制的主入口函数"""
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y, _, _ = _layer_norm_fwd_impl(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )
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
    
    def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        return rms_norm_gated_impl(x, self.weight, z=z, eps=self.eps)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""
    
    def __init__(self, hidden_size: int = 1024, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
        return rms_norm_gated(x, self.weight, bias=None, z=z, eps=self.eps)


def get_inputs():
    """生成测试输入"""
    batch_size = 4
    seq_len = 128
    hidden_size = 1024
    
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda")
    z = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda")
    
    return [x, z]


def get_init_inputs():
    """获取初始化参数"""
    return [1024, 1e-6]  # hidden_size, eps

