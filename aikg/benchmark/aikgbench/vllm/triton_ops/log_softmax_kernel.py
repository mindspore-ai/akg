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

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/batch_invariant.py
# vLLM函数: log_softmax
# 功能: 沿最后一维计算log_softmax（Triton实现）
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _log_softmax_kernel_impl(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    
    沿最后一维计算log_softmax的Triton kernel。
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax_impl(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)

    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel_impl[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        沿指定维度计算log_softmax
        
        Args:
            input: 输入张量 [..., N]
            
        Returns:
            log_softmax结果
        """
        return log_softmax_impl(input, self.dim)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.batch_invariant import log_softmax

        return log_softmax(input, self.dim)


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    dtype = torch.float16
    
    # 典型场景：batch_size=32, seq_len=128, vocab_size=32000
    batch_size = 32
    seq_len = 128
    vocab_size = 32000
    
    # logits输入
    input = torch.randn(batch_size, seq_len, vocab_size, dtype=dtype, device=device)
    
    return [input]


def get_init_inputs():
    """生成初始化参数"""
    return [-1]  # dim

