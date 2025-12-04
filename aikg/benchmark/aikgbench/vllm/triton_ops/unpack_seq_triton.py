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
# vLLM函数: unpack_seq_triton
# 功能: 将批处理张量解包回原始序列格式
# 测试文件: tests/kernels/attention/test_pack_unpack_triton.py
# ============================================================================


@triton.jit
def _unpack_seq_triton_kernel_impl(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    """
    Unpack a packed tensor back to the original sequence format.
    
    将批处理张量解包回原始序列格式的Triton kernel。
    """
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # bounds: compute start from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    # compute output row indices for valid (b, t)
    out_row = in_start + off_t

    # Pointers
    # packed_ptr: row-major [B, Lmax, D]
    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # out_ptr: row-major [N, D]
    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    # Load from packed tensor and store to output
    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)


def unpack_seq_triton_impl(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Efficient Triton implementation.

    Args:
        packed_tensor: [B, Lmax, ...] - packed tensor from pack_seq_triton
        lengths: [B] - sequence lengths for each batch
        block_t: block size for time dimension
        block_d: block size for feature dimension

    Returns:
        unpacked_tensor: [N, ...] where N = sum(lengths)
    """

    # Handle multi-dimensional input by reshaping to (B, Lmax, -1)
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    # Calculate total number of elements
    N = int(lengths.sum().item())

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _unpack_seq_triton_kernel_impl[grid](
        packed_reshaped,
        out,
        lengths.int(),
        B,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    # Reshape output back to original dimensions (except first dimension)
    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(
        self,
        block_t: int = 64,
        block_d: int = 64,
    ):
        super().__init__()
        self.block_t = block_t
        self.block_d = block_d

    def forward(
        self,
        packed_tensor: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unpack a packed tensor back to the original sequence format.
        
        Args:
            packed_tensor: [B, Lmax, H, D] - packed tensor
            lengths: [B] - sequence lengths for each batch
            
        Returns:
            unpacked: [N, H, D] - unpacked tensor where N = sum(lengths)
        """
        return unpack_seq_triton_impl(packed_tensor, lengths, self.block_t, self.block_d)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(
        self,
        block_t: int = 64,
        block_d: int = 64,
    ):
        super().__init__()
        self.block_t = block_t
        self.block_d = block_d

    def forward(
        self,
        packed_tensor: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.attention.ops.common import unpack_seq_triton

        return unpack_seq_triton(packed_tensor, lengths, self.block_t, self.block_d)


def get_inputs():
    """生成测试输入"""
    # 基于测试文件: tests/kernels/attention/test_pack_unpack_triton.py
    # 使用测试用例: test_pack_unpack_roundtrip_fp8 中的一个典型case
    
    device = "cuda"
    dtype = torch.float16  # 使用float16而不是float8以便更广泛兼容
    
    # B=4, Lmax=5, H=16, D=32, lengths=[5, 5, 5, 5]
    B = 4
    Lmax = 5
    H = 16
    D = 32
    lengths_list = [5, 5, 5, 5]
    
    # 创建已打包的输入张量 [B, Lmax, H, D]
    packed_tensor = torch.randn(B, Lmax, H, D, dtype=dtype, device=device)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
    
    return [packed_tensor, lengths]


def get_init_inputs():
    """生成初始化参数"""
    return [
        64,  # block_t
        64,  # block_d
    ]

