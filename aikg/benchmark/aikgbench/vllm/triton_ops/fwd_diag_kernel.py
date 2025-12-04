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
# 源文件: vllm/model_executor/layers/lightning_attn.py
# vLLM函数: lightning_attention (内部kernel)
# 功能: Lightning Attention对角块计算
# 测试文件: tests/kernels/attention/test_lightning_attn.py
# ============================================================================


@triton.jit
def _fwd_diag_kernel_impl(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    CBLOCK: tl.constexpr,
):
    """
    Lightning Attention对角块计算kernel
    
    计算attention矩阵的对角块，每个对角块表示query关注同一块中的key。
    """
    # This kernel computes the diagonal blocks of the attention matrix
    # Each diagonal block represents attention where queries attend to keys in the same block
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK  # batch-head index
    off_block = off % NUM_BLOCK  # block index within the sequence
    off_cblock = tl.program_id(1)  # sub-block index within a block

    off_h = off_bh % h  # head index

    # Calculate base offsets for the current batch and head
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    # Calculate offsets for the current block
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    # Calculate offsets for the current sub-block
    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    # Calculate pointers to the query, key, value, and output tensors
    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    # Load query values
    q = tl.load(Q_block_ptr, mask=block_offset + q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    # Initialize output accumulator
    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    # Process all sub-blocks up to and including the current one (causal attention)
    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        # Apply causal mask: only attend to positions before the current one
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        # Load key and value
        k_trans = tl.load(
            K_trans_block_ptr,
            mask=block_offset + kv_index[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr,
            mask=block_offset + kv_index[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        # Compute attention scores and apply decay
        qk = tl.dot(q, k_trans) * decay

        # Compute weighted values and accumulate
        qkv += tl.dot(qk, v)

        # Move to the next sub-block
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    # Store the result
    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=block_offset + q_index[:, None] < n,
    )


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(
        self,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        block_size: int = 64,
        cblock_size: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.block_size = block_size
        self.cblock_size = cblock_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lightning Attention对角块计算
        
        Args:
            query: [batch, num_heads, seq_len, head_dim_qk]
            key: [batch, num_heads, seq_len, head_dim_qk]
            value: [batch, num_heads, seq_len, head_dim_v]
            s: [num_heads] - 衰减率
            
        Returns:
            output: [batch, num_heads, seq_len, head_dim_v]
        """
        b, h, n, d = query.shape
        e = value.shape[-1]
        
        output = torch.empty_like(value)
        
        NUM_BLOCK = triton.cdiv(n, self.block_size)
        NUM_CBLOCK = triton.cdiv(self.block_size, self.cblock_size)
        
        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        
        _fwd_diag_kernel_impl[grid](
            query,
            key,
            value,
            output,
            s,
            b=b,
            h=h,
            n=n,
            d=d,
            e=e,
            BLOCK=self.block_size,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=self.cblock_size,
        )
        
        return output


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(
        self,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        block_size: int = 64,
        cblock_size: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.block_size = block_size
        self.cblock_size = cblock_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.lightning_attn import _fwd_diag_kernel

        b, h, n, d = query.shape
        e = value.shape[-1]
        
        output = torch.empty_like(value)
        
        NUM_BLOCK = triton.cdiv(n, self.block_size)
        NUM_CBLOCK = triton.cdiv(self.block_size, self.cblock_size)
        
        grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
        
        _fwd_diag_kernel[grid](
            query,
            key,
            value,
            output,
            s,
            b=b,
            h=h,
            n=n,
            d=d,
            e=e,
            BLOCK=self.block_size,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=self.cblock_size,
        )
        
        return output


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    dtype = torch.float16
    
    # 基于tests/kernels/attention/test_lightning_attn.py
    batch_size = 2
    num_heads = 8
    seq_len = 256
    head_dim_qk = 64
    head_dim_v = 64
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim_qk, dtype=dtype, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim_qk, dtype=dtype, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim_v, dtype=dtype, device=device)
    s = torch.randn(num_heads, dtype=torch.float32, device=device).abs()  # 衰减率为正
    
    return [query, key, value, s]


def get_init_inputs():
    """生成初始化参数"""
    return [
        8,   # num_heads
        64,  # head_dim_qk
        64,  # head_dim_v
        64,  # block_size
        32,  # cblock_size
    ]

