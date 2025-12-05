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
# vLLM函数: linear_decode_forward_triton
# 功能: Lightning Attention线性解码kernel
# 测试文件: tests/kernels/attention/test_lightning_attn.py
# ============================================================================


@triton.jit
def _linear_attn_decode_kernel_impl(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_cache_ptr,
    slope_rate,
    slot_idx,
    output_ptr,
    D: tl.constexpr,
    qkv_b_stride,
    qkv_h_stride,
    cache_b_stride,
    cache_h_stride,
    cache_d0_stride,
    cache_d1_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    线性attention解码kernel，使用KV缓存
    
    为单个token计算attention，使用KV缓存。
    """
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # head index
    pid_d = tl.program_id(2)  # dimension block index

    # Load slot index for the current batch
    slot_id = tl.load(slot_idx + pid_b).to(tl.int64)

    # Skip if slot_id is -1 (padding)
    if slot_id == -1:
        return

    batch_id = pid_b
    head_id = pid_h

    # Load decay rate for the current head
    ratio = tl.load(slope_rate + pid_h)

    # Calculate offsets for dimensions
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
    cache_d_offsets = (
        qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[None, :] * cache_d1_stride
    )

    # Calculate offsets for the current batch and head
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride

    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride

    # Create masks for loading tensors
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D

    # Load query, key, and value tensors
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)

    # Compute key-value outer product
    kv_outer = k[:, None] * v[None, :]
    kv_mask = qk_mask[:, None] & v_mask[None, :]

    # Apply decay to previous KV cache
    ratio = tl.exp(-ratio)
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_outer = kv_outer + ratio * kv_cache_old

    # Compute attention output
    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)

    # Update KV cache and store output
    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, head_dim: int = 64, block_size: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        slot_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        线性attention解码
        
        Args:
            q: [batch, num_heads, 1, head_dim]
            k: [batch, num_heads, 1, head_dim]
            v: [batch, num_heads, 1, head_dim]
            kv_caches: [num_slots, num_heads, head_dim, head_dim] - KV缓存
            slope_rate: [num_heads] - 衰减率
            slot_idx: [batch] - slot索引
            
        Returns:
            output: [batch, num_heads, 1, head_dim]
        """
        batch_size, num_heads, _, head_dim = q.shape
        num_d_blocks = triton.cdiv(head_dim, self.block_size)
        
        # 压缩为3维来调用kernel
        q_3d = q.squeeze(2)
        k_3d = k.squeeze(2)
        v_3d = v.squeeze(2)
        output_3d = torch.zeros_like(q_3d)
        
        grid = (batch_size, num_heads, num_d_blocks)
        
        _linear_attn_decode_kernel_impl[grid](
            q_3d,
            k_3d,
            v_3d,
            kv_caches,
            slope_rate,
            slot_idx,
            output_3d,
            D=head_dim,
            qkv_b_stride=q_3d.stride(0),
            qkv_h_stride=q_3d.stride(1),
            cache_b_stride=kv_caches.stride(0),
            cache_h_stride=kv_caches.stride(1),
            cache_d0_stride=kv_caches.stride(2),
            cache_d1_stride=kv_caches.stride(3),
            BLOCK_SIZE=self.block_size,
        )
        
        # 恢复为4维输出
        return output_3d.unsqueeze(2)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, head_dim: int = 64, block_size: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        slot_idx: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.lightning_attn import (
            linear_decode_forward_triton,
        )

        return linear_decode_forward_triton(
            q, k, v, kv_caches, slope_rate, slot_idx, self.block_size
        )


def get_inputs():
    """生成测试输入"""
    dtype = torch.float16
    
    batch_size = 4
    num_heads = 8
    head_dim = 64
    num_slots = 128
    
    # vLLM期望q, k, v是4维的 [B, H, 1, D]
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype)
    kv_caches = torch.randn(num_slots, num_heads, head_dim, head_dim, dtype=dtype)
    slope_rate = torch.randn(num_heads, dtype=torch.float32).abs()
    slot_idx = torch.randint(0, num_slots, (batch_size,), dtype=torch.long)
    
    return [q, k, v, kv_caches, slope_rate, slot_idx]


def get_init_inputs():
    """生成初始化参数"""
    return [64, 32]

