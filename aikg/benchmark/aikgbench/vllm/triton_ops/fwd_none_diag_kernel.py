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
# 功能: Lightning Attention非对角块计算
# 测试文件: tests/kernels/attention/test_lightning_attn.py
# ============================================================================


@triton.jit
def _fwd_none_diag_kernel_impl(
    Q,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    E_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    """
    Lightning Attention非对角块计算kernel
    
    计算attention矩阵的非对角块，query关注不同块中的key。
    """
    # This kernel computes the non-diagonal blocks of the attention matrix
    # Each non-diagonal block represents attention where queries attend to keys in different blocks
    off_bh = tl.program_id(0)  # batch-head index
    off_h = off_bh % h  # head index

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK  # block index
    off_c = off_nc % NUM_CBLOCK  # sub-block index
    off_e = tl.program_id(2)  # output feature block index

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    block_offset = n_offset + c_offset

    # Calculate offsets for the current batch, head, and block
    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset
    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    # Calculate pointers to the query, output, and key-value tensors
    Q_block_ptr = (
        Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    # Load the key-value outer product for the current block
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_index = block_offset + tl.arange(0, CBLOCK)

    # Load query values
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # Compute decay factors for the current sub-block
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))

    # Compute non-diagonal attention output
    qkv_none_diag = tl.dot(q, kv) * q_decay

    # Load diagonal attention output (computed by _fwd_diag_kernel)
    qkv_diag = tl.load(O_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # Combine diagonal and non-diagonal attention outputs
    qkv = qkv_diag + qkv_none_diag

    # Store the result
    tl.store(
        O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n
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
        e_fblock: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.block_size = block_size
        self.cblock_size = cblock_size
        self.e_fblock = e_fblock

    def forward(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        s: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lightning Attention非对角块计算
        
        Args:
            query: [batch, num_heads, seq_len, head_dim_qk]
            output: [batch, num_heads, seq_len, head_dim_v] - 对角块的输出
            s: [num_heads] - 衰减率
            kv: [batch, num_heads, num_blocks, head_dim_qk, head_dim_v]
            
        Returns:
            output: 更新后的输出（原地修改）
        """
        b, h, n, d = query.shape
        e = self.head_dim_v
        NUM_BLOCK = kv.shape[2]
        NUM_CBLOCK = triton.cdiv(self.block_size, self.cblock_size)
        NUM_E_FBLOCK = triton.cdiv(e, self.e_fblock)
        
        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_E_FBLOCK)
        
        _fwd_none_diag_kernel_impl[grid](
            query,
            output,
            s,
            kv,
            b=b,
            h=h,
            n=n,
            d=d,
            e=e,
            BLOCK=self.block_size,
            NUM_BLOCK=NUM_BLOCK,
            E_FBLOCK=self.e_fblock,
            CBLOCK=self.cblock_size,
            NUM_CBLOCK=NUM_CBLOCK,
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
        e_fblock: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.block_size = block_size
        self.cblock_size = cblock_size
        self.e_fblock = e_fblock

    def forward(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        s: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.lightning_attn import _fwd_none_diag_kernel

        b, h, n, d = query.shape
        e = self.head_dim_v
        NUM_BLOCK = kv.shape[2]
        NUM_CBLOCK = triton.cdiv(self.block_size, self.cblock_size)
        NUM_E_FBLOCK = triton.cdiv(e, self.e_fblock)
        
        grid = (b * h, NUM_BLOCK * NUM_CBLOCK, NUM_E_FBLOCK)
        
        _fwd_none_diag_kernel[grid](
            query,
            output,
            s,
            kv,
            b=b,
            h=h,
            n=n,
            d=d,
            e=e,
            BLOCK=self.block_size,
            NUM_BLOCK=NUM_BLOCK,
            E_FBLOCK=self.e_fblock,
            CBLOCK=self.cblock_size,
            NUM_CBLOCK=NUM_CBLOCK,
        )
        
        return output


def get_inputs():
    """生成测试输入"""
    
    dtype = torch.float16
    
    batch_size = 2
    num_heads = 8
    seq_len = 256
    head_dim_qk = 64
    head_dim_v = 64
    block_size = 64
    num_blocks = triton.cdiv(seq_len, block_size)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim_qk, dtype=dtype)
    output = torch.randn(batch_size, num_heads, seq_len, head_dim_v, dtype=dtype)
    s = torch.randn(num_heads, dtype=torch.float32).abs()
    kv = torch.randn(batch_size, num_heads, num_blocks, head_dim_qk, head_dim_v, dtype=dtype)
    
    return [query, output, s, kv]


def get_init_inputs():
    """生成初始化参数"""
    return [8, 64, 64, 64, 32, 64]

