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
import math

# ============================================================================
# Attention Kernel参考信息
# ============================================================================
# 源文件: thirdparty/attention_kernel_triton/decode_flash_attention_redundant_var_len_paged.py
# 算子名称: Decode Flash Attention (Variable Length Paged)
# 功能: 支持变长序列和Paged KV Cache的Decode Flash Attention
# 特性:
#   - 支持Paged KV Cache存储
#   - 支持批次内不同序列长度
#   - 支持Grouped Query Attention (GQA)
#   - Query redundant优化（复制16次）
# 性能: 相比PyTorch reference实现44x加速
# ============================================================================


# ============================================================================
# Triton Kernel实现
# ============================================================================

@triton.jit
def inner_kernel(
    Q_block,
    K_block_ptr,
    V_block_ptr,
    KV_SEQ_BLOCK_SIZE,
    tmp_O_block,
    tmp_m_i,
    tmp_l_i,
    SEQ_LEN,
    softmax_scale,
    KV_ranges,
):
    seq_lo, seq_hi = 0, SEQ_LEN
    
    for start_kv in range(seq_lo, seq_hi, KV_SEQ_BLOCK_SIZE):
        start_kv = tl.multiple_of(start_kv, KV_SEQ_BLOCK_SIZE)

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        # 变长mask：只关注实际序列长度内的tokens
        mask = SEQ_LEN > KV_ranges + start_kv
        QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)[None, :]
        
        m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1))
        QK_block -= m_ij[:, None]

        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)

        alpha = tl.math.exp(tmp_m_i - m_ij)
        tmp_l_i = tmp_l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float32)
        tmp_O_block = tmp_O_block * alpha[:, None] 
        tmp_O_block = tl.dot(P_block, V_block, tmp_O_block)

        tmp_m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (KV_SEQ_BLOCK_SIZE, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, KV_SEQ_BLOCK_SIZE))
        
    return tmp_O_block, tmp_m_i, tmp_l_i


@triton.jit
def flash_attention_paged_kernel(
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    kv_last_page_len_ptr,
    O_ptr,
    softmax_scale,
    stride_q_B,
    stride_q_H_GQA,
    stride_q_1,
    stride_q_D,
    stride_paged_kv_cache_B,
    stride_paged_kv_cache_2,
    stride_paged_kv_cache_H,
    stride_paged_kv_cache_page,
    stride_paged_kv_cache_D,
    stride_O_B,
    stride_O_H_GQA,
    stride_O_1,
    stride_O_D,
    BATCH_SIZE,
    H_GQA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_group_size: tl.constexpr,
    page_size: tl.constexpr,
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
    redundant_len: tl.constexpr,
):
    """
    q: (B, H*GQA_group_size, redundant_len, D)
    paged_kv_cache: (all_num_pages, 2, H, page_size, D) (2 for k and v)
    kv_page_indptr: (B+1) (int32)
    kv_page_indices: (total_num_pages) (int32)
    kv_last_page_len: (B) (int32)
    O: (B, H*GQA_group_size, redundant_len, D)
    """
    
    B_H_GQA_id = tl.program_id(0)
    out_batch_id = B_H_GQA_id // H_GQA
    out_head_id = B_H_GQA_id % H_GQA 
    kv_head_id = out_head_id // GQA_group_size

    indptr_start = tl.load(kv_page_indptr_ptr + out_batch_id)
    indptr_end = tl.load(kv_page_indptr_ptr + out_batch_id + 1)
    num_pages = indptr_end - indptr_start

    qo_offset = out_batch_id * stride_q_B + out_head_id * stride_q_H_GQA
    qo_offset_O = out_batch_id * stride_O_B + out_head_id * stride_O_H_GQA

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset_O,
        shape = (redundant_len, HEAD_DIM),
        strides = (stride_O_1, stride_O_D),
        block_shape = (redundant_len, HEAD_DIM),
        offsets = (0, 0),
        order = (1, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        base = q_ptr + qo_offset,
        shape = (redundant_len, HEAD_DIM),
        strides = (stride_q_1, stride_q_D),
        block_shape = (redundant_len, HEAD_DIM),
        offsets = (0, 0),
        order = (1, 0),
    )

    q_block = tl.load(q_block_ptr)
    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)
    
    tmp_O_block = tl.zeros((redundant_len, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((redundant_len,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((redundant_len,), dtype=tl.float32)

    # 遍历所有pages
    for i in range(num_pages):
        page_idx = tl.load(kv_page_indices_ptr + indptr_start + i)
        
        if i == num_pages - 1:
            last_page_len = tl.load(kv_last_page_len_ptr + out_batch_id)
        else:
            last_page_len = page_size

        k_ptr_offset = page_idx * stride_paged_kv_cache_B + kv_head_id * stride_paged_kv_cache_H
        v_ptr_offset = page_idx * stride_paged_kv_cache_B + stride_paged_kv_cache_2 + kv_head_id * stride_paged_kv_cache_H

        K_block_ptr = tl.make_block_ptr(
            base = paged_kv_cache_ptr + k_ptr_offset,
            shape = (HEAD_DIM, page_size),
            strides = (stride_paged_kv_cache_D, stride_paged_kv_cache_page),
            block_shape = (HEAD_DIM, KV_SEQ_BLOCK_SIZE),
            offsets = (0, 0),
            order = (0, 1),
        )
        
        V_block_ptr = tl.make_block_ptr(
            base = paged_kv_cache_ptr + v_ptr_offset,
            shape = (page_size, HEAD_DIM),
            strides = (stride_paged_kv_cache_page, stride_paged_kv_cache_D),
            block_shape = (KV_SEQ_BLOCK_SIZE, HEAD_DIM),
            offsets = (0, 0),
            order = (1, 0),
        )

        tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
            q_block, K_block_ptr, V_block_ptr,
            KV_SEQ_BLOCK_SIZE, tmp_O_block, tmp_m_i, tmp_l_i,
            last_page_len, softmax_scale, KV_ranges,
        )

    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """Triton实现的Paged Decode Flash Attention"""
    
    def __init__(
        self,
        kv_seq_blk_size: int = 16,
        redundant_len: int = 16,
    ):
        super().__init__()
        self.kv_seq_blk_size = kv_seq_blk_size
        self.redundant_len = redundant_len
    
    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim]
            paged_kv_cache: [all_num_pages, 2, num_heads_kv, page_size, head_dim]
            kv_page_indptr: [batch+1] (int32)
            kv_page_indices: [total_num_pages] (int32)
            kv_last_page_len: [batch] (int32)
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        q_redundant = q.repeat_interleave(self.redundant_len, dim=2)
        
        B, H_GQA, _, D = q_redundant.shape
        H = paged_kv_cache.shape[2]
        GQA_group_size = H_GQA // H
        page_size = paged_kv_cache.shape[3]
        
        softmax_scale = float(1.0 / math.sqrt(D))
        
        O = torch.empty((B, H_GQA, self.redundant_len, D), dtype=q.dtype, device=q.device)
        
        grid = (B * H_GQA, 1, 1)
        
        flash_attention_paged_kernel[grid](
            q_redundant, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len,
            O, softmax_scale,
            q_redundant.stride(0), q_redundant.stride(1), q_redundant.stride(2), q_redundant.stride(3),
            paged_kv_cache.stride(0), paged_kv_cache.stride(1), paged_kv_cache.stride(2), 
            paged_kv_cache.stride(3), paged_kv_cache.stride(4),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, H_GQA, D, GQA_group_size, page_size,
            self.kv_seq_blk_size, self.redundant_len,
        )
        
        return O[:, :, 0, :].contiguous().unsqueeze(2)


class ModelTorch(nn.Module):
    """PyTorch原生实现的Paged Decode Flash Attention"""
    
    def __init__(
        self,
        kv_seq_blk_size: int = 16,
        redundant_len: int = 16,
    ):
        super().__init__()
        self.kv_seq_blk_size = kv_seq_blk_size
        self.redundant_len = redundant_len
    
    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim]
            paged_kv_cache: [all_num_pages, 2, num_heads_kv, page_size, head_dim]
            kv_page_indptr: [batch+1] (int32)
            kv_page_indices: [total_num_pages] (int32)
            kv_last_page_len: [batch] (int32)
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[3]
        num_kv_heads = paged_kv_cache.shape[2]
        
        k_cache, v_cache = paged_kv_cache[:, 0], paged_kv_cache[:, 1]
        
        outputs = []
        for b in range(batch_size):
            page_start = kv_page_indptr[b].item()
            page_end = kv_page_indptr[b + 1].item()
            
            if page_start == page_end:
                outputs.append(torch.zeros_like(q[b:b+1]))
                continue
            
            seq_pages = kv_page_indices[page_start:page_end]
            seq_k_parts = []
            seq_v_parts = []
            
            for i, page_idx in enumerate(seq_pages):
                k_page = k_cache[page_idx]
                v_page = v_cache[page_idx]
                
                if i == len(seq_pages) - 1:
                    last_len = kv_last_page_len[b].item()
                    k_page = k_page[:, :last_len, :]
                    v_page = v_page[:, :last_len, :]
                
                seq_k_parts.append(k_page)
                seq_v_parts.append(v_page)
            
            seq_k = torch.cat(seq_k_parts, dim=1)
            seq_v = torch.cat(seq_v_parts, dim=1)
            
            if num_qo_heads != num_kv_heads:
                groups = num_qo_heads // num_kv_heads
                seq_k = seq_k.repeat_interleave(groups, dim=0)
                seq_v = seq_v.repeat_interleave(groups, dim=0)
            
            q_b = q[b]
            scores = torch.matmul(q_b, seq_k.permute(0, 2, 1)) / math.sqrt(head_dim)
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, seq_v)
            outputs.append(out.unsqueeze(0))
        
        return torch.cat(outputs, dim=0)


def get_inputs():
    """生成测试输入"""
    torch.manual_seed(42)
    
    device = "cuda"
    dtype = torch.float32
    num_kv_heads = 2
    gqa_group_size = 2
    num_qo_heads = num_kv_heads * gqa_group_size
    head_dim = 64
    page_size = 64
    max_num_pages = 16
    batch_size = 3
    
    paged_kv_cache = torch.randn(
        max_num_pages, 2, num_kv_heads, page_size, head_dim,
        device=device, dtype=dtype
    )
    
    kv_page_indptr = torch.tensor([0, 4, 5, 8], dtype=torch.int32, device=device)
    kv_page_indices = torch.tensor([0, 1, 3, 5, 2, 6, 7, 4], dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([2, 22, 3], dtype=torch.int32, device=device)
    
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, device=device, dtype=dtype)
    
    return [q, paged_kv_cache, kv_page_indptr, kv_page_indices, kv_last_page_len]


def get_init_inputs():
    """生成初始化参数"""
    return [16, 16]  # kv_seq_blk_size, redundant_len

