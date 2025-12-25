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
# 源文件: thirdparty/attention_kernel_triton/sparsetoken_decode_flash_attention_redundant_var_len_paged.py
# 算子名称: Sparse Token Decode Flash Attention (Paged)
# 功能: 结合Sparse Token和Paged KV Cache的Decode Flash Attention
# 特性:
#   - 支持稀疏token选择
#   - 支持Paged KV Cache存储
#   - 支持Grouped Query Attention (GQA)
#   - Query redundant优化（复制16次）
#   - 适用于长序列+稀疏关注场景
# ============================================================================


# ============================================================================
# Triton Kernel实现
# ============================================================================

@triton.jit
def sparsetoken_flash_attention_decode_paged_kernel(
    q_ptr,
    paged_kv_cache_ptr,
    kv_page_indptr_ptr,
    kv_page_indices_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    O_ptr,
    softmax_scale,
    stride_paged_kv_cache_pages,
    stride_paged_kv_cache_2,
    stride_paged_kv_cache_H,
    stride_paged_kv_cache_P,
    stride_paged_kv_cache_D,
    stride_q_B,
    stride_q_H_GQA,
    stride_q_1,
    stride_q_D,
    stride_sparse_ind_B,
    stride_sparse_ind_H,
    stride_sparse_nnz_B,
    stride_sparse_nnz_H,
    stride_O_B,
    stride_O_H_GQA,
    stride_O_1,
    stride_O_D,
    BATCH_SIZE,
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_groups_size: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
    redundant_len: tl.constexpr,
):
    """
    q: [B, qo_heads, redundant_len, head_dim]
    paged_kv_cache: [all_pages, 2, kv_heads, page_size, head_dim]
    kv_page_indptr: [B+1]
    kv_page_indices: [total_pages]
    sparse_ind: [B, H, L_max]
    sparse_nnz: [B, H, 1]
    O: [B, qo_heads, redundant_len, head_dim]
    """
    
    B_H_id = tl.program_id(0)
    out_batch_id = B_H_id // NUM_QO_HEADS
    out_head_id = B_H_id % NUM_QO_HEADS
    kv_head_id = out_head_id // GQA_groups_size

    qo_offset = out_batch_id * stride_q_B + out_head_id * stride_q_H_GQA

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
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
    tmp_O_block = tl.zeros((redundant_len, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((redundant_len,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((redundant_len,), dtype=tl.float32)

    b_h_nzz = tl.load(sparse_nnz_ptr + out_batch_id * stride_sparse_nnz_B + out_head_id * stride_sparse_nnz_H)
    b_h_ind_ptr_base = sparse_ind_ptr + out_batch_id * stride_sparse_ind_B + out_head_id * stride_sparse_ind_H

    page_idx_start = tl.load(kv_page_indptr_ptr + out_batch_id)

    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)
    
    # 遍历sparse选中的tokens，从paged KV cache中加载
    for ind_start_idx in range(0, b_h_nzz, KV_SEQ_BLOCK_SIZE):
        mask = ind_start_idx + KV_ranges < b_h_nzz
        
        token_idx = tl.load(b_h_ind_ptr_base + ind_start_idx + KV_ranges, mask=mask, other=0)
        page_idx = token_idx // PAGE_SIZE
        offset_in_page = token_idx % PAGE_SIZE
        page_id = tl.load(kv_page_indices_ptr + page_idx_start + page_idx, mask=mask, other=0)

        k_ptr = paged_kv_cache_ptr + page_id * stride_paged_kv_cache_pages + \
                kv_head_id * stride_paged_kv_cache_H + \
                offset_in_page * stride_paged_kv_cache_P
        
        v_ptr = paged_kv_cache_ptr + page_id * stride_paged_kv_cache_pages + \
                1 * stride_paged_kv_cache_2 + \
                kv_head_id * stride_paged_kv_cache_H + \
                offset_in_page * stride_paged_kv_cache_P

        shared_K = tl.load(
            k_ptr[None,:] + tl.arange(0, HEAD_DIM)[:, None] * stride_paged_kv_cache_D,
            mask=mask[None, :],
            other=0.0
        )

        shared_V = tl.load(
            v_ptr[:,None] + tl.arange(0, HEAD_DIM)[None, :] * stride_paged_kv_cache_D,
            mask=mask[:, None],
            other=0.0
        )

        # 计算attention
        QK_block = tl.dot(q_block, shared_K)
        mask = ind_start_idx + KV_ranges < b_h_nzz
        QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)[None, :]
        
        m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1))
        QK_block -= m_ij[:, None]

        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)
    
        alpha = tl.math.exp(tmp_m_i - m_ij)
        tmp_l_i = tmp_l_i * alpha + l_ij

        P_block = P_block.to(tl.float32)
        tmp_O_block = tmp_O_block * alpha[:, None] 
        tmp_O_block = tl.dot(P_block, shared_V, tmp_O_block)

        tmp_m_i = m_ij

    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """Triton实现的Sparse Token Paged Decode Flash Attention"""
    
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
        sparse_ind: torch.Tensor,
        sparse_nnz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim]
            paged_kv_cache: [all_pages, 2, num_heads_kv, page_size, head_dim]
            kv_page_indptr: [batch+1] (int32)
            kv_page_indices: [total_pages] (int32)
            sparse_ind: [batch, num_heads, L_max] - sparse token indices
            sparse_nnz: [batch, num_heads] - actual number of sparse tokens
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        q_redundant = q.repeat_interleave(self.redundant_len, dim=2)
        
        B, num_qo_heads, _, head_dim = q_redundant.shape
        num_kv_heads = paged_kv_cache.shape[2]
        groups = num_qo_heads // num_kv_heads
        page_size = paged_kv_cache.shape[3]
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        O = torch.zeros((B, num_qo_heads, self.redundant_len, head_dim), 
                       device=q.device, dtype=q.dtype)
        
        grid = (B * num_qo_heads, 1, 1)
        
        sparsetoken_flash_attention_decode_paged_kernel[grid](
            q_redundant, paged_kv_cache, kv_page_indptr, kv_page_indices, 
            sparse_ind, sparse_nnz, O, softmax_scale,
            paged_kv_cache.stride(0), paged_kv_cache.stride(1), paged_kv_cache.stride(2),
            paged_kv_cache.stride(3), paged_kv_cache.stride(4),
            q_redundant.stride(0), q_redundant.stride(1), q_redundant.stride(2), q_redundant.stride(3),
            sparse_ind.stride(0), sparse_ind.stride(1),
            sparse_nnz.stride(0), sparse_nnz.stride(1),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, num_qo_heads, head_dim, groups, page_size,
            self.kv_seq_blk_size, self.redundant_len,
        )
        
        return O[:, :, 0, :].contiguous().unsqueeze(2)


class ModelTorch(nn.Module):
    """PyTorch原生实现的Sparse Token Paged Decode Flash Attention"""
    
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
        sparse_ind: torch.Tensor,
        sparse_nnz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim]
            paged_kv_cache: [all_pages, 2, num_heads_kv, page_size, head_dim]
            kv_page_indptr: [batch+1] (int32)
            kv_page_indices: [total_pages] (int32)
            sparse_ind: [batch, num_heads, L_max] - sparse token indices
            sparse_nnz: [batch, num_heads] - actual number of sparse tokens
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        batch_size = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[3]
        num_kv_heads = paged_kv_cache.shape[2]
        page_size = paged_kv_cache.shape[3]
        groups = num_qo_heads // num_kv_heads
        
        k_cache, v_cache = paged_kv_cache[:, 0], paged_kv_cache[:, 1]
        
        outputs = []
        for b in range(batch_size):
            batch_outputs = []
            for h in range(num_qo_heads):
                nnz = sparse_nnz[b, h].item()
                if nnz == 0:
                    batch_outputs.append(torch.zeros((1, head_dim), device=q.device, dtype=q.dtype))
                    continue
                
                seq_head_k_parts = []
                seq_head_v_parts = []
                indices = sparse_ind[b, h, :nnz]
                kv_h = h // groups
                
                for i in range(nnz):
                    token_idx = indices[i].item()
                    page_idx = token_idx // page_size
                    page_id = kv_page_indices[kv_page_indptr[b].item() + page_idx]
                    offset_in_page = token_idx % page_size
                    seq_head_k_parts.append(k_cache[page_id, kv_h, offset_in_page:offset_in_page+1, :])
                    seq_head_v_parts.append(v_cache[page_id, kv_h, offset_in_page:offset_in_page+1, :])
                
                seq_head_k = torch.cat(seq_head_k_parts, dim=0).unsqueeze(0)
                seq_head_v = torch.cat(seq_head_v_parts, dim=0).unsqueeze(0)
                
                q_bh = q[b, h:h+1]
                scores = torch.matmul(q_bh, seq_head_k.permute(0, 2, 1)) / math.sqrt(head_dim)
                weights = torch.softmax(scores, dim=-1)
                out = torch.matmul(weights, seq_head_v)
                batch_outputs.append(out.squeeze(1))
            
            outputs.append(torch.cat(batch_outputs, dim=0).unsqueeze(0))
        
        return torch.cat(outputs, dim=0).unsqueeze(2)


def get_inputs():
    """生成测试输入"""
    torch.manual_seed(42)
    
    dtype = torch.float32
    num_kv_heads = 8
    gqa_group_size = 4
    num_qo_heads = num_kv_heads * gqa_group_size
    head_dim = 256
    page_size = 256
    max_num_pages = 1024
    batch_size = 2
    kept_ratio = 0.02
    
    paged_kv_cache = torch.randn(
        max_num_pages, 2, num_kv_heads, page_size, head_dim,
        dtype=dtype
    )
    
    kv_page_indptr = torch.tensor([0, 3, 7], dtype=torch.int32)
    kv_page_indices = torch.tensor([0, 1, 2, 5, 6, 7, 8], dtype=torch.int32)
    
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, dtype=dtype)
    
    # 生成sparse indices
    total_tokens = page_size * 4  # 假设每个batch有4个pages的tokens
    L_max = int(total_tokens * (kept_ratio + 0.1))
    sparse_nnz = torch.zeros((batch_size, num_qo_heads), dtype=torch.int32)
    sparse_ind = torch.zeros((batch_size, num_qo_heads, L_max), dtype=torch.int32) - 1
    
    for b in range(batch_size):
        for h in range(num_qo_heads):
            sample_prob = torch.rand((total_tokens,))
            kept_mask = sample_prob < kept_ratio
            kept_nnz = kept_mask.sum().item()
            if kept_nnz == 0:
                kept_nnz = 1
                kept_mask[0] = True
            sparse_nnz[b, h] = kept_nnz
            sparse_ind[b, h, :kept_nnz] = torch.nonzero(kept_mask, as_tuple=False).squeeze(-1)
    
    return [q, paged_kv_cache, kv_page_indptr, kv_page_indices, sparse_ind, sparse_nnz]


def get_init_inputs():
    """生成初始化参数"""
    return [16, 16]  # kv_seq_blk_size, redundant_len

