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
# 功能: Lightning Attention KV归约
# 测试文件: tests/kernels/attention/test_lightning_attn.py
# ============================================================================


@triton.jit
def _fwd_kv_reduce_impl(
    S,
    KV,
    KV_HISTORY,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
):
    """
    Lightning Attention KV归约kernel
    
    跨块归约key-value外积并更新KV历史。
    """
    # This kernel reduces the key-value outer products across blocks and updates the KV history
    off_bh = tl.program_id(0)  # batch-head index
    off_h = off_bh % h  # head index

    kv_offset = off_bh * NUM_BLOCK * d * e

    # Calculate pointer to the key-value tensor
    KV_block_ptr = (
        KV
        + kv_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the decay rate for the current head
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # Calculate pointer to the key-value history tensor
    kv_history_offset = off_bh * d * e
    KV_HISTORY_block_ptr = (
        KV_HISTORY
        + kv_history_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the previous key-value history
    kv_pre = tl.load(KV_HISTORY_block_ptr).to(tl.float32)

    # Process all blocks in reverse order to compute the prefix sum
    for i in range(NUM_BLOCK):
        block_size = min(n - i * BLOCK, BLOCK)
        # Compute decay factor for the current block
        block_decay = tl.exp(-s.to(tl.float32) * block_size)

        # Load the current key-value outer product
        kv_cur = tl.load(KV_block_ptr).to(tl.float32)
        # Store the previous key-value history to the current block
        tl.store(KV_block_ptr, kv_pre.to(KV_block_ptr.dtype.element_ty))

        # Update the key-value history with the current block
        kv_pre = block_decay * kv_pre + kv_cur
        KV_block_ptr += d * e

    # Store the updated key-value history
    tl.store(KV_HISTORY_block_ptr, kv_pre)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(
        self,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        block_size: int = 64,
        d_fblock: int = 64,
        e_fblock: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.block_size = block_size
        self.d_fblock = d_fblock
        self.e_fblock = e_fblock

    def forward(
        self,
        s: torch.Tensor,
        kv: torch.Tensor,
        kv_history: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Lightning Attention KV归约
        
        Args:
            s: [num_heads] - 衰减率
            kv: [batch, num_heads, num_blocks, head_dim_qk, head_dim_v] - KV外积
            kv_history: [batch, num_heads, head_dim_qk, head_dim_v] - KV历史
            seq_len: 序列长度
            
        Returns:
            kv_history: 更新后的KV历史（原地修改kv和kv_history）
        """
        b, h = kv.shape[0], kv.shape[1]
        NUM_BLOCK = kv.shape[2]
        d = self.head_dim_qk
        e = self.head_dim_v
        
        grid = (b * h,)
        
        _fwd_kv_reduce_impl[grid](
            s,
            kv,
            kv_history,
            b=b,
            h=h,
            n=seq_len,
            d=d,
            e=e,
            BLOCK=self.block_size,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=self.d_fblock,
            E_FBLOCK=self.e_fblock,
        )
        
        return kv_history


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(
        self,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        block_size: int = 64,
        d_fblock: int = 64,
        e_fblock: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.block_size = block_size
        self.d_fblock = d_fblock
        self.e_fblock = e_fblock

    def forward(
        self,
        s: torch.Tensor,
        kv: torch.Tensor,
        kv_history: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.lightning_attn import _fwd_kv_reduce

        b, h = kv.shape[0], kv.shape[1]
        NUM_BLOCK = kv.shape[2]
        d = self.head_dim_qk
        e = self.head_dim_v
        
        grid = (b * h,)
        
        _fwd_kv_reduce[grid](
            s,
            kv,
            kv_history,
            b=b,
            h=h,
            n=seq_len,
            d=d,
            e=e,
            BLOCK=self.block_size,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=self.d_fblock,
            E_FBLOCK=self.e_fblock,
        )
        
        return kv_history


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    dtype = torch.float16
    
    batch_size = 2
    num_heads = 8
    seq_len = 256
    head_dim_qk = 64
    head_dim_v = 64
    block_size = 64
    num_blocks = triton.cdiv(seq_len, block_size)
    
    s = torch.randn(num_heads, dtype=torch.float32, device=device).abs()
    kv = torch.randn(batch_size, num_heads, num_blocks, head_dim_qk, head_dim_v, dtype=dtype, device=device)
    kv_history = torch.randn(batch_size, num_heads, head_dim_qk, head_dim_v, dtype=dtype, device=device)
    
    return [s, kv, kv_history, seq_len]


def get_init_inputs():
    """生成初始化参数"""
    return [8, 64, 64, 64, 64, 64]

