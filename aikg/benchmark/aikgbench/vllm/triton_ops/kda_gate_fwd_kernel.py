# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/fla/ops/kda.py
# vLLM函数: fused_kda_gate
# 功能: KDA门控前向传播（Key-Delta Attention）
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def log(x):
    """对数函数"""
    return tl.log(x)


@triton.jit
def kda_gate_fwd_kernel_impl(
    g,
    A,
    y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    KDA门控前向kernel
    
    应用softplus激活和缩放：y = A * softplus(g + bias, beta)
    """
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = -tl.exp(b_a)

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias[None, :]

    # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
    # When beta * x > threshold, use linear approximation x
    # Use threshold to switch to linear when beta*x > threshold
    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
    b_y = b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate_impl(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """KDA门控的Python包装函数"""
    orig_shape = g.shape[:-1]
    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    H = A.shape[0]
    D = head_k_dim
    
    # 重塑g为[T, H, D]
    g = g.view(T, H, D)
    y = torch.empty_like(g)
    
    BT = 32
    BD = min(triton.next_power_of_2(D), 128)
    
    grid = (triton.cdiv(T, BT), H)
    
    kda_gate_fwd_kernel_impl[grid](
        g,
        A,
        y,
        g_bias,
        beta=beta,
        threshold=threshold,
        T=T,
        H=H,
        D=D,
        BT=BT,
        BD=BD,
        HAS_BIAS=g_bias is not None,
    )
    
    # 恢复原始形状
    return y.view(*orig_shape, H, D)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(
        self,
        num_heads: int,
        head_k_dim: int,
        beta: float = 1.0,
        threshold: float = 20.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.beta = beta
        self.threshold = threshold

    def forward(
        self,
        g: torch.Tensor,
        A: torch.Tensor,
        g_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        KDA门控前向传播
        
        Args:
            g: [..., H*D] - 门控输入
            A: [H] 或 [1, 1, H, 1] - 缩放参数
            g_bias: [H*D] 可选 - 偏置
            
        Returns:
            output: [..., H, D] - 门控输出
        """
        # 确保A是[H]形状
        if A.ndim > 1:
            A = A.squeeze()
        
        return fused_kda_gate_impl(
            g, A, self.head_k_dim, g_bias, self.beta, self.threshold
        )


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(
        self,
        num_heads: int,
        head_k_dim: int,
        beta: float = 1.0,
        threshold: float = 20.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.beta = beta
        self.threshold = threshold

    def forward(
        self,
        g: torch.Tensor,
        A: torch.Tensor,
        g_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.kda import fused_kda_gate

        # 确保A是[H]形状
        if A.ndim > 1:
            A = A.squeeze()
        
        return fused_kda_gate(
            g, A, self.head_k_dim, g_bias, self.beta, self.threshold
        )


def get_inputs():
    """生成测试输入"""
    
    dtype = torch.float32
    
    batch_size = 8
    seq_len = 64
    num_heads = 4
    head_k_dim = 64
    
    g = torch.randn(batch_size, seq_len, num_heads * head_k_dim, dtype=dtype)
    A = torch.randn(num_heads, dtype=dtype)
    g_bias = torch.randn(num_heads * head_k_dim, dtype=dtype)
    
    return [g, A, g_bias]


def get_init_inputs():
    """生成初始化参数"""
    return [4, 64, 1.0, 20.0]

