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
# 源文件: vllm/model_executor/layers/batch_invariant.py
# vLLM函数: mean_dim
# 功能: 沿单个维度计算均值（Triton实现）
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def mean_kernel_impl(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    
    沿单个维度计算均值的Triton kernel。
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim_impl(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert -input.ndim <= dim < input.ndim

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Create output shape
    if keepdim:
        output_shape = shape[:dim] + [1] + shape[dim + 1 :]
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Allocate output
    output = torch.empty(output_shape, device=input.device, dtype=dtype)

    # If output is empty, return it
    if output.numel() == 0:
        return output

    # Get strides
    strides = list(input.stride())

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (M * K,)

    mean_kernel_impl[grid](
        input,
        output.reshape(M, K) if not keepdim else output.reshape(M, 1, K).squeeze(1),
        strides[dim - 1] if dim > 0 else input.numel() // shape[0] if len(shape) > 1 else 0,
        strides[dim],
        strides[dim + 1] if dim < len(shape) - 1 else 1,
        K if not keepdim else 1,
        1 if not keepdim else K,
        M,
        N,
        K,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, dim: int = -1, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        沿指定维度计算均值
        
        Args:
            input: 输入张量
            
        Returns:
            均值结果
        """
        return mean_dim_impl(input, self.dim, self.keepdim)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, dim: int = -1, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.batch_invariant import mean_dim

        return mean_dim(input, self.dim, self.keepdim)


def get_inputs():
    """生成测试输入"""
    
    dtype = torch.float16
    
    # 测试场景：计算序列维度的均值
    batch_size = 8
    seq_len = 256
    hidden_size = 1024
    
    input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    
    return [input]


def get_init_inputs():
    """生成初始化参数"""
    return [1, False]  # dim=1 (seq_len维度), keepdim=False

