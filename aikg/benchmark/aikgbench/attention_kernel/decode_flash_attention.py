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
# 源文件: thirdparty/attention_kernel_triton/decode_flash_attention_redundant.py
# 算子名称: Decode Flash Attention (Redundant)
# 功能: Decode阶段的Flash Attention实现
# 特性:
#   - Query长度固定为1 (通过redundant技巧优化为16)
#   - 支持Grouped Query Attention (GQA)
#   - KV cache长度可变
#   - 无causal mask（decode阶段不需要）
# 性能: 相比PyTorch reference实现6.8x加速
# 优化技巧: Query redundant - 将单个query复制16次提高计算效率
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
):
    seq_lo, seq_hi = 0, SEQ_LEN

    V_block_ptr = tl.advance(V_block_ptr, (seq_lo, 0))
    K_block_ptr = tl.advance(K_block_ptr, (0, seq_lo))

    for start_kv in range(seq_lo, seq_hi, KV_SEQ_BLOCK_SIZE):
        start_kv = tl.multiple_of(start_kv, KV_SEQ_BLOCK_SIZE)

        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        m_ij = tl.maximum(tmp_m_i, tl.max(QK_block, axis=1) * softmax_scale )
        QK_block = QK_block * softmax_scale - m_ij[:, None]
        
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
def flash_attention_kernel(
    q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    softmax_scale,
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
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_group_size: tl.constexpr, 
    KV_SEQ_BLOCK_SIZE: tl.constexpr,
):
    B_H_pid = tl.program_id(0)
    batch_id = B_H_pid // NUM_HEADS
    head_id = B_H_pid % NUM_HEADS

    qo_offset = batch_id * stride_Q_batch + head_id * stride_Q_head
    kv_offset = batch_id * stride_K_batch + (head_id // GQA_group_size) * stride_K_head 

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qo_offset,
        shape = (16, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        block_shape = (16, HEAD_DIM),
        offsets = (0, 0),
        order = (1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        base = q_ptr + qo_offset,
        shape = (16, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim),
        block_shape = (16, HEAD_DIM),
        offsets = (0, 0),
        order = (1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + kv_offset,
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (stride_K_dim, stride_K_seq),
        block_shape = (HEAD_DIM, KV_SEQ_BLOCK_SIZE),
        offsets = (0, 0),
        order = (0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base = V_ptr + kv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        block_shape = (KV_SEQ_BLOCK_SIZE, HEAD_DIM),
        offsets = (0, 0),
        order = (1, 0),
    )

    tmp_O_block = tl.zeros((16, HEAD_DIM), dtype=tl.float32)
    tmp_m_i = tl.zeros((16,), dtype=tl.float32) - float('inf')
    tmp_l_i = tl.zeros((16,), dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr)

    tmp_O_block, tmp_m_i, tmp_l_i = inner_kernel(
        Q_block, K_block_ptr, V_block_ptr,
        KV_SEQ_BLOCK_SIZE,
        tmp_O_block, tmp_m_i, tmp_l_i,
        SEQ_LEN, softmax_scale,
    )

    tmp_O_block = tmp_O_block / tmp_l_i[:, None]
    tl.store(O_block_ptr, tmp_O_block.to(O_ptr.type.element_ty))


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """Triton实现的Decode Flash Attention"""
    
    def __init__(
        self,
        kv_seq_blk_size: int = 64,
        q_redundant_len: int = 16,
    ):
        super().__init__()
        self.kv_seq_blk_size = kv_seq_blk_size
        self.q_redundant_len = q_redundant_len
    
    def forward(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        gqa_group_size: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim] - decode阶段单个query
            K: [batch, num_heads_kv, seq_len, head_dim] - KV cache
            V: [batch, num_heads_kv, seq_len, head_dim] - KV cache
            gqa_group_size: GQA分组大小
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        # 使用redundant技巧：将q复制16次提高计算效率
        q_redundant = q.repeat(1, 1, self.q_redundant_len, 1)
        
        B, H_kv, S, D = K.shape
        softmax_scale = 1.0 / (D ** 0.5)
        O = torch.empty((B, H_kv * gqa_group_size, self.q_redundant_len, D), 
                       dtype=q.dtype, device=q.device)
        
        grid = (B * H_kv * gqa_group_size, 1, 1)
        
        flash_attention_kernel[grid](
            q_redundant, K, V, O, softmax_scale,
            q_redundant.stride(0), q_redundant.stride(1), q_redundant.stride(2), q_redundant.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            BATCH_SIZE=B, NUM_HEADS=H_kv * gqa_group_size, SEQ_LEN=S, HEAD_DIM=D,
            GQA_group_size=gqa_group_size,
            KV_SEQ_BLOCK_SIZE=self.kv_seq_blk_size,
        )
        
        # 只返回第一个token的结果
        return O[:, :, :1, :].contiguous()


class ModelTorch(nn.Module):
    """PyTorch原生实现的Decode Flash Attention"""
    
    def __init__(
        self,
        kv_seq_blk_size: int = 64,
        q_redundant_len: int = 16,
    ):
        super().__init__()
        self.kv_seq_blk_size = kv_seq_blk_size
        self.q_redundant_len = q_redundant_len
    
    def forward(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        gqa_group_size: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, 1, head_dim] - decode阶段单个query
            K: [batch, num_heads_kv, seq_len, head_dim] - KV cache
            V: [batch, num_heads_kv, seq_len, head_dim] - KV cache
            gqa_group_size: GQA分组大小
        Returns:
            O: [batch, num_heads, 1, head_dim]
        """
        B, H_q, _, D = q.shape
        _, H_kv, S, _ = K.shape
        softmax_scale = 1.0 / (D ** 0.5)
        
        # 处理GQA
        if gqa_group_size > 1:
            K = K.repeat_interleave(gqa_group_size, dim=1)
            V = V.repeat_interleave(gqa_group_size, dim=1)
        
        # 计算attention scores
        P = torch.matmul(q, K.transpose(-2, -1)) * softmax_scale
        
        # Softmax
        P = torch.softmax(P.float(), dim=-1).to(q.dtype)
        
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
    dtype = torch.float32
    device = "cuda"
    
    q = torch.randn(
        batch_size, num_heads_kv * gqa_group_size, 1, head_dim,
        dtype=dtype, device=device
    )
    K = torch.randn(
        batch_size, num_heads_kv, seq_len, head_dim,
        dtype=dtype, device=device
    )
    V = torch.randn(
        batch_size, num_heads_kv, seq_len, head_dim,
        dtype=dtype, device=device
    )
    
    return [q, K, V, gqa_group_size]


def get_init_inputs():
    """生成初始化参数"""
    return [64, 16]  # kv_seq_blk_size, q_redundant_len

