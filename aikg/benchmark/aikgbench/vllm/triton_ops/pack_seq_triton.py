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
# 源文件: vllm/attention/ops/common.py
# vLLM函数: pack_seq_triton
# 功能: 将不同长度的序列打包成批处理张量
# 测试文件: tests/kernels/attention/test_pack_unpack_triton.py
# ============================================================================


@triton.jit
def _pack_seq_kernel_impl(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    """
    Pack sequences of different lengths into a batched tensor.
    
    将不同长度的序列打包成批处理张量的Triton kernel。
    """
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Compute start index and sequence length from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax

    # compute input row indices for valid (b, t)
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask

    # Pointers
    # x_ptr: row-major [N, D]
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

    # out_ptr: row-major [B, Lmax, D]
    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # Initialize with PAD (cast will occur as needed based on out_ptr dtype)
    d_mask = off_d[None, :] < D
    pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

    # Load & write only where within seq_len
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)


def pack_seq_triton_impl(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Pack sequences of different lengths into a batched tensor.

    Args:
        x: [N, ...] - input tensor where N is total number of tokens
        lengths: [B] - sequence lengths for each batch
        pad_value: value to use for padding
        block_t: block size for time dimension
        block_d: block size for feature dimension

    Returns:
        packed: [B, Lmax, ...] - packed tensor
    """

    # Handle multi-dimensional input by reshaping to (N, -1)
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())

    # Starts are computed inside the kernel from lengths

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _pack_seq_kernel_impl[grid](
        x_reshaped,
        out,
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=float(pad_value),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    # Reshape output back to original dimensions (except first dimension)
    if len(original_shape) > 2:
        output_shape = (B, Lmax) + original_shape[1:]
        out = out.reshape(output_shape)

    return out


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(
        self,
        pad_value: float = -float("inf"),
        block_t: int = 64,
        block_d: int = 64,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.block_t = block_t
        self.block_d = block_d

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pack sequences of different lengths into a batched tensor.
        
        Args:
            x: [N, H, D] - input tensor where N is total number of tokens
            lengths: [B] - sequence lengths for each batch
            
        Returns:
            packed: [B, Lmax, H, D] - packed tensor
        """
        return pack_seq_triton_impl(
            x, lengths, self.pad_value, self.block_t, self.block_d
        )


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(
        self,
        pad_value: float = -float("inf"),
        block_t: int = 64,
        block_d: int = 64,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.block_t = block_t
        self.block_d = block_d

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.attention.ops.common import pack_seq_triton

        return pack_seq_triton(x, lengths, self.pad_value, self.block_t, self.block_d)


def get_inputs():
    """生成测试输入"""
    # 基于测试文件: tests/kernels/attention/test_pack_unpack_triton.py
    # 使用测试用例: test_pack_seq_basic_fp8 中的一个典型case
    
    device = "cuda"
    dtype = torch.float16  # 使用float16而不是float8以便更广泛兼容
    
    # N=20, H=16, D=32, B=4, lengths=[5, 5, 5, 5]
    N = 20
    H = 16
    D = 32
    lengths_list = [5, 5, 5, 5]
    
    # 创建输入张量 [N, H, D]
    x = torch.randn(N, H, D, dtype=dtype, device=device)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
    
    return [x, lengths]


def get_init_inputs():
    """生成初始化参数"""
    return [
        -float("inf"),  # pad_value (默认值)
        64,  # block_t
        64,  # block_d
    ]

