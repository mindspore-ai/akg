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
# Attention Kernel参考信息
# ============================================================================
# 源文件: thirdparty/attention_kernel_triton/sparsetoken_decode_flash_attention_redundant.py
# 算子名称: Sparse Token Decode Flash Attention
# 功能: 稀疏token选择的Decode Flash Attention
# 特性:
#   - 只关注重要的sparse tokens而非所有tokens
#   - 支持Grouped Query Attention (GQA)
#   - Query redundant优化（复制16次）
#   - 适用于长序列场景，大幅降低计算量
# 性能: 相比PyTorch reference实现72x加速，相比mask实现42x加速
# ============================================================================


# ============================================================================
# Triton Kernel实现
# ============================================================================

@triton.jit
def sparsetoken_flash_attention_decode_kernel(
    q_ptr, 
    K_ptr,
    V_ptr,
    sparse_ind_ptr,
    sparse_nnz_ptr,
    O_ptr,
    softmax_scale,
    q_stride_B, q_stride_H, q_stride_1, q_stride_D,
    K_stride_B, K_stride_H, K_stride_S, K_stride_D,
    V_stride_B, V_stride_H, V_stride_S, V_stride_D,
    sparse_ind_stride_B, sparse_ind_stride_H, 
    sparse_nnz_stride_B, sparse_nnz_stride_H,
    O_stride_B, O_stride_H, O_stride_1, O_stride_D,
    B,
    num_qo_heads: tl.constexpr,
    head_dim: tl.constexpr,
    GQA_group_size: tl.constexpr,
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
    redundant_len: tl.constexpr,
):
    """
    q: (B, qo_heads, redundant_len, head_dim)
    K: (B, kv_heads, SEQ_LEN, head_dim)
    V: (B, kv_heads, SEQ_LEN, head_dim)
    sparse_ind: (B, qo_heads, L_max)  (padded with -1)
    sparse_nnz: (B, qo_heads)  (actual lengths)
    """
    
    B_H_pid = tl.program_id(0)
    out_batch_id = B_H_pid // num_qo_heads
    out_head_id = B_H_pid % num_qo_heads
    kv_head_id = out_head_id // GQA_group_size

    qo_offset = out_batch_id * q_stride_B + out_head_id * q_stride_H

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
        shape = (redundant_len, head_dim),
        strides = (O_stride_1, O_stride_D),
        block_shape = (redundant_len, head_dim),
        offsets = (0, 0),
        order = (1, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        base = q_ptr + qo_offset,
        shape = (redundant_len, head_dim),
        strides = (q_stride_1, q_stride_D),
        block_shape = (redundant_len, head_dim),
        offsets = (0, 0),
        order = (1, 0),
    )

    q_block = tl.load(q_block_ptr)
    tmp_O_block = tl.zeros((redundant_len, head_dim), dtype=tl.float32)
    tmp_m_i = tl.zeros((redundant_len,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((redundant_len,), dtype=tl.float32)

    b_h_nzz = tl.load(sparse_nnz_ptr + out_batch_id * sparse_nnz_stride_B + out_head_id * sparse_nnz_stride_H)
    b_h_ind_ptr_base = sparse_ind_ptr + out_batch_id * sparse_ind_stride_B + out_head_id * sparse_ind_stride_H

    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)
    
    # 只遍历sparse选中的tokens
    for ind_start_idx in range(0, b_h_nzz, KV_SEQ_BLOCK_SIZE):
        mask = ind_start_idx + KV_ranges < b_h_nzz
        token_idx = tl.load(b_h_ind_ptr_base + ind_start_idx + KV_ranges, mask=mask, other=0)
        
        k_ptr = K_ptr + out_batch_id * K_stride_B + kv_head_id * K_stride_H + token_idx * K_stride_S
        v_ptr = V_ptr + out_batch_id * V_stride_B + kv_head_id * V_stride_H + token_idx * V_stride_S

        # 加载sparse选中的K和V
        shared_K = tl.load(
            k_ptr[None,:] + tl.arange(0, head_dim)[:, None] * K_stride_D,
            mask=mask[None, :],
            other=0.0
        )

        shared_V = tl.load(
            v_ptr[:,None] + tl.arange(0, head_dim)[None, :] * V_stride_D,
            mask=mask[:, None],
            other=0.0
        )

        # 计算attention
        QK_block = tl.dot(q_block, shared_K)
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
    """Triton实现的Sparse Token Decode Flash Attention"""
    
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
        K: torch.Tensor,
        V: torch.Tensor,
        sparse_ind: torch.Tensor,
        sparse_nnz: torch.Tensor,
        gqa_group_size: int,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim]
            K: [batch, num_heads_kv, seq_len, head_dim]
            V: [batch, num_heads_kv, seq_len, head_dim]
            sparse_ind: [batch, num_heads, L_max] - sparse token indices
            sparse_nnz: [batch, num_heads] - actual number of sparse tokens
            gqa_group_size: GQA分组大小
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        q_redundant = q.repeat_interleave(self.redundant_len, dim=2)
        
        B, num_qo_heads, _, head_dim = q_redundant.shape
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        O = torch.zeros((B, num_qo_heads, self.redundant_len, head_dim), 
                       device=q.device, dtype=q.dtype)
        
        grid = (B * num_qo_heads, 1, 1)
        
        sparsetoken_flash_attention_decode_kernel[grid](
            q_redundant, K, V, sparse_ind, sparse_nnz, O, softmax_scale,
            q_redundant.stride(0), q_redundant.stride(1), q_redundant.stride(2), q_redundant.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            sparse_ind.stride(0), sparse_ind.stride(1),
            sparse_nnz.stride(0), sparse_nnz.stride(1),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, num_qo_heads, head_dim, gqa_group_size,
            self.kv_seq_blk_size, self.redundant_len,
        )
        
        return O[:, :, :1, :].contiguous()


class ModelTorch(nn.Module):
    """PyTorch原生实现的Sparse Token Decode Flash Attention"""
    
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
        K: torch.Tensor,
        V: torch.Tensor,
        sparse_ind: torch.Tensor,
        sparse_nnz: torch.Tensor,
        gqa_group_size: int,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim]
            K: [batch, num_heads_kv, seq_len, head_dim]
            V: [batch, num_heads_kv, seq_len, head_dim]
            sparse_ind: [batch, num_heads, L_max] - sparse token indices
            sparse_nnz: [batch, num_heads] - actual number of sparse tokens
            gqa_group_size: GQA分组大小
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        B, qo_heads, _, head_dim = q.shape
        softmax_scale = 1.0 / (head_dim ** 0.5)
        output = torch.zeros((B, qo_heads, 1, head_dim), dtype=q.dtype, device=q.device)
        
        for b in range(B):
            for h in range(qo_heads):
                k_h = h // gqa_group_size
                q_vec = q[b, h, 0]
                nnz = sparse_nnz[b, h].item()
                if nnz == 0:
                    nnz = 1
                k_indices = sparse_ind[b, h, :nnz]
                k_vecs = K[b, k_h, k_indices]
                v_vecs = V[b, k_h, k_indices]
                attn_scores = torch.matmul(k_vecs, q_vec) * softmax_scale
                attn_probs = torch.softmax(attn_scores, dim=0)
                out_vec = torch.matmul(attn_probs.unsqueeze(0), v_vecs)
                output[b, h, 0] = out_vec
        
        return output


def get_inputs():
    """生成测试输入"""
    torch.manual_seed(42)
    
    dtype = torch.float32
    num_kv_heads = 8
    gqa_group_size = 4
    num_qo_heads = num_kv_heads * gqa_group_size
    head_dim = 128
    batch_size = 1
    seq_len = 32000
    kept_ratio = 0.02
    L_max = int(seq_len * (kept_ratio + 0.1))
    
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, dtype=dtype)
    K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)
    V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)
    
    # 生成sparse indices
    sparse_nnz = torch.zeros((batch_size, num_qo_heads, 1), dtype=torch.int32)
    sparse_ind = torch.zeros((batch_size, num_qo_heads, L_max), dtype=torch.int32) - 1
    
    for b in range(batch_size):
        for h in range(num_qo_heads):
            sample_prob = torch.rand((seq_len,))
            kept_mask = sample_prob < kept_ratio
            kept_nnz = kept_mask.sum().item()
            if kept_nnz == 0:
                kept_nnz = 1
                kept_mask[0] = True
            sparse_nnz[b, h, 0] = kept_nnz
            sparse_ind[b, h, :kept_nnz] = torch.nonzero(kept_mask, as_tuple=False).squeeze(-1)
    
    sparse_nnz = sparse_nnz.squeeze(-1)  # [batch, num_heads]
    
    return [q, K, V, sparse_ind, sparse_nnz, gqa_group_size]


def get_init_inputs():
    """生成初始化参数"""
    return [16, 16]  # kv_seq_blk_size, redundant_len

