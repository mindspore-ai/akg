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
# 源文件: thirdparty/attention_kernel_triton/prefill_flash_attention.py
# 算子名称: Prefill Flash Attention
# 功能: Flash Attention的prefill阶段实现
# 特性:
#   - 支持causal和non-causal attention
#   - 支持Grouped Query Attention (GQA)
#   - 批次内序列长度相同
#   - 仅支持forward推理
# 性能: 相比PyTorch reference实现45x加速
# ============================================================================


# ============================================================================
# Triton Kernel实现
# ============================================================================

@triton.jit
def inner_kernel(
    # query block
    Q_block,
    # K V ptrs
    K_block_ptr,
    V_block_ptr,
    # block size
    QO_SEQ_BLOCK_SIZE,
    KV_SEQ_BLOCK_SIZE,
    # block id
    seq_blk_id,
    # locals
    tmp_O_block,
    tmp_m_i,
    tmp_l_i,
    # stage
    STAGE,
    # ranges
    Q_ranges,
    KV_ranges,
    SEQ_LEN,
    # other
    softmax_scale,
):
    if STAGE == 1:
        # from 0 to the left of the diagonal; non-causal part in causal attn
        seq_lo, seq_hi = 0, seq_blk_id * QO_SEQ_BLOCK_SIZE
    elif STAGE == 2:
        # from the left of the diagonal to the remaining
        seq_lo, seq_hi = seq_blk_id * QO_SEQ_BLOCK_SIZE, (seq_blk_id + 1) * QO_SEQ_BLOCK_SIZE
        seq_lo = tl.multiple_of(seq_lo, QO_SEQ_BLOCK_SIZE)
    else:
        # for non-causal 
        seq_lo, seq_hi = 0, SEQ_LEN

    V_block_ptr = tl.advance(V_block_ptr, (seq_lo, 0))
    K_block_ptr = tl.advance(K_block_ptr, (0, seq_lo))

    # loop K V by KV_SEQ_BLOCK_SIZE
    for start_kv in range(seq_lo, seq_hi, KV_SEQ_BLOCK_SIZE):
        start_kv = tl.multiple_of(start_kv, KV_SEQ_BLOCK_SIZE)

        # compute q@k
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        # handle causal
        if STAGE == 2:
            mask = Q_ranges[:, None] >= KV_ranges[None, :] + start_kv
            QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)
            # mantain the max value 
            m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1) * softmax_scale )
            QK_block = QK_block * softmax_scale - m_ij[:, None]
        
        # compute exp, sumofexp, 
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)

        # record alpha for sumofexp correction, and correct sumofexp
        alpha = tl.math.exp(tmp_m_i - m_ij)
        tmp_l_i = tmp_l_i * alpha + l_ij

        # compute output
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float32)
        tmp_O_block = tmp_O_block * alpha[:, None] 
        tmp_O_block = tl.dot(P_block, V_block, tmp_O_block)

        tmp_m_i = m_ij

        # advance the loop
        V_block_ptr = tl.advance(V_block_ptr, (KV_SEQ_BLOCK_SIZE, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, KV_SEQ_BLOCK_SIZE))

    return tmp_O_block, tmp_m_i, tmp_l_i


@triton.jit
def flash_attention_kernel(
    # data ptr
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    softmax_scale,
    # stride
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    # shapes
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_group_size: tl.constexpr, 
    # block
    QO_SEQ_BLOCK_SIZE: tl.constexpr,
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
    # stage
    STAGE: tl.constexpr,
):
    tl.static_assert(QO_SEQ_BLOCK_SIZE <= HEAD_DIM)

    B_H_pid = tl.program_id(0)
    batch_id = B_H_pid // NUM_HEADS
    head_id = B_H_pid % NUM_HEADS

    S_pid = tl.program_id(1)
    seq_blk_id = S_pid

    qo_offset = batch_id * stride_Q_batch + head_id * stride_Q_head
    kv_offset = batch_id * stride_K_batch + (head_id // GQA_group_size) * stride_K_head 

    # O block
    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        block_shape = (QO_SEQ_BLOCK_SIZE, HEAD_DIM),
        offsets = (seq_blk_id*QO_SEQ_BLOCK_SIZE, 0),
        order = (1, 0),
    )

    # Q block
    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + qo_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim),
        block_shape = (QO_SEQ_BLOCK_SIZE, HEAD_DIM),
        offsets = (seq_blk_id*QO_SEQ_BLOCK_SIZE, 0),
        order = (1, 0),
    )

    # K block
    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + kv_offset,
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (stride_K_dim, stride_K_seq),
        block_shape = (HEAD_DIM, KV_SEQ_BLOCK_SIZE),
        offsets = (0, 0),
        order = (0, 1),
    )

    # V block
    V_block_ptr = tl.make_block_ptr(
        base = V_ptr + kv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        block_shape = (KV_SEQ_BLOCK_SIZE, HEAD_DIM),
        offsets = (0, 0),
        order = (1, 0),
    )

    # local O block and other local intermediate values
    tmp_O_block = tl.zeros((QO_SEQ_BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((QO_SEQ_BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((QO_SEQ_BLOCK_SIZE,), dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # comput the ranges for threads in program
    Q_ranges = tl.arange(0, QO_SEQ_BLOCK_SIZE) + seq_blk_id * QO_SEQ_BLOCK_SIZE
    KV_ranges = tl.arange(0, KV_SEQ_BLOCK_SIZE)

    # call programs 
    if STAGE == 1 or STAGE == 3:
        tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
            Q_block, K_block_ptr, V_block_ptr,
            QO_SEQ_BLOCK_SIZE, KV_SEQ_BLOCK_SIZE,
            seq_blk_id, tmp_O_block, tmp_m_i, tmp_l_i,
            4 - STAGE, Q_ranges, KV_ranges, SEQ_LEN,
            softmax_scale,
        )

    if STAGE == 3:
        tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
            Q_block, K_block_ptr, V_block_ptr,
            QO_SEQ_BLOCK_SIZE, KV_SEQ_BLOCK_SIZE,
            seq_blk_id, tmp_O_block, tmp_m_i, tmp_l_i,
            2, Q_ranges, KV_ranges, SEQ_LEN,
            softmax_scale,
        )

    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """Triton实现的Flash Attention Prefill"""
    
    def __init__(
        self,
        q_seq_blk_size: int = 64,
        kv_seq_blk_size: int = 16,
    ):
        super().__init__()
        self.q_seq_blk_size = q_seq_blk_size
        self.kv_seq_blk_size = kv_seq_blk_size
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool,
        gqa_group_size: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            Q: [batch, num_heads, seq_len, head_dim]
            K: [batch, num_heads_kv, seq_len, head_dim]
            V: [batch, num_heads_kv, seq_len, head_dim]
            causal: 是否使用causal mask
            gqa_group_size: GQA分组大小
        Returns:
            O: [batch, num_heads, seq_len, head_dim]
        """
        B, H, S, D = Q.shape
        softmax_scale = 1.0 / (D ** 0.5)
        O = torch.empty_like(Q)
        stage = 3 if causal else 1
        
        grid = (B * H, triton.cdiv(S, self.q_seq_blk_size), 1)
        
        flash_attention_kernel[grid](
            Q, K, V, O, softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            BATCH_SIZE=B, NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D,
            GQA_group_size=gqa_group_size,
            QO_SEQ_BLOCK_SIZE=self.q_seq_blk_size,
            KV_SEQ_BLOCK_SIZE=self.kv_seq_blk_size,
            STAGE=stage,
        )
        
        return O


class ModelTorch(nn.Module):
    """PyTorch原生实现的Flash Attention Prefill"""
    
    def __init__(
        self,
        q_seq_blk_size: int = 64,
        kv_seq_blk_size: int = 16,
    ):
        super().__init__()
        self.q_seq_blk_size = q_seq_blk_size
        self.kv_seq_blk_size = kv_seq_blk_size
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool,
        gqa_group_size: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            Q: [batch, num_heads, seq_len, head_dim]
            K: [batch, num_heads_kv, seq_len, head_dim]
            V: [batch, num_heads_kv, seq_len, head_dim]
            causal: 是否使用causal mask
            gqa_group_size: GQA分组大小
        Returns:
            O: [batch, num_heads, seq_len, head_dim]
        """
        B, H, S, D = Q.shape
        softmax_scale = 1.0 / (D ** 0.5)
        
        # 处理GQA
        if gqa_group_size > 1:
            K = K.repeat_interleave(gqa_group_size, dim=1)
            V = V.repeat_interleave(gqa_group_size, dim=1)
        
        # 计算attention scores
        P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
        
        # 应用causal mask
        if causal:
            mask = torch.tril(torch.ones((S, S), device=Q.device, dtype=torch.bool))
            P = P.masked_fill(~mask, float('-inf'))
        
        # Softmax
        P = torch.softmax(P.float(), dim=-1).to(Q.dtype)
        
        # 计算输出
        O = torch.matmul(P, V)
        
        return O


def get_inputs():
    """生成测试输入"""
    torch.manual_seed(42)
    
    batch_size = 8
    num_heads_kv = 16
    seq_len = 1024
    head_dim = 64
    gqa_group_size = 4
    causal = True
    dtype = torch.float32
    
    Q = torch.randn(
        batch_size, num_heads_kv * gqa_group_size, seq_len, head_dim,
        dtype=dtype
    )
    K = torch.randn(
        batch_size, num_heads_kv, seq_len, head_dim,
        dtype=dtype
    )
    V = torch.randn(
        batch_size, num_heads_kv, seq_len, head_dim,
        dtype=dtype
    )
    
    return [Q, K, V, causal, gqa_group_size]


def get_init_inputs():
    """生成初始化参数"""
    return [64, 16]  # q_seq_blk_size, kv_seq_blk_size

