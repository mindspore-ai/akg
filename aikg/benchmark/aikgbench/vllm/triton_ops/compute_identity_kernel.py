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
# 源文件: vllm/model_executor/layers/fused_moe/fused_moe.py
# vLLM函数: zero_experts_compute_triton (内部调用)
# 功能: 计算恒等专家输出（MoE中的zero expert处理）
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def compute_identity_kernel_impl(
    top_k: int,
    hidden_states_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    num_tokens: int,
    output_ptr: tl.tensor,
    hidden_dim: int,
    scales_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    计算恒等专家输出kernel
    
    对于MoE中的zero expert，计算hidden_states * expert_scales的加权和。
    """
    pid = tl.program_id(0)

    batch_id = pid // (hidden_dim // BLOCK_SIZE)
    dim_offset = pid % (hidden_dim // BLOCK_SIZE) * BLOCK_SIZE

    if batch_id >= num_tokens or dim_offset >= hidden_dim:
        return

    h = tl.load(
        hidden_states_ptr
        + batch_id * hidden_dim
        + dim_offset
        + tl.arange(0, BLOCK_SIZE),
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(top_k):
        scale = tl.load(expert_scales_ptr + batch_id * scales_stride + i)
        result += h * scale

    tl.store(
        output_ptr + batch_id * hidden_dim + dim_offset + tl.arange(0, BLOCK_SIZE),
        result,
        mask=(dim_offset + tl.arange(0, BLOCK_SIZE)) < hidden_dim,
    )


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, block_size: int = 128):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算恒等专家输出
        
        Args:
            hidden_states: [num_tokens, hidden_dim] - 隐藏状态
            expert_scales: [num_tokens, top_k] - 专家权重
            
        Returns:
            output: [num_tokens, hidden_dim] - 加权后的输出
        """
        num_tokens, hidden_dim = hidden_states.shape
        top_k = expert_scales.shape[1]
        
        output = torch.zeros_like(hidden_states)
        
        num_blocks = triton.cdiv(num_tokens * hidden_dim, self.block_size)
        grid = (num_blocks,)
        
        compute_identity_kernel_impl[grid](
            top_k=top_k,
            hidden_states_ptr=hidden_states,
            expert_scales_ptr=expert_scales,
            num_tokens=num_tokens,
            output_ptr=output,
            hidden_dim=hidden_dim,
            scales_stride=expert_scales.stride(0),
            BLOCK_SIZE=self.block_size,
        )
        
        return output


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, block_size: int = 128):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_scales: torch.Tensor,
    ) -> torch.Tensor:
        # vLLM的compute_identity_kernel不直接导出
        # 我们需要通过zero_experts_compute_triton调用
        # 为了简化，这里直接调用kernel
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            compute_identity_kernel,
        )

        num_tokens, hidden_dim = hidden_states.shape
        top_k = expert_scales.shape[1]
        
        output = torch.zeros_like(hidden_states)
        
        num_blocks = triton.cdiv(num_tokens * hidden_dim, self.block_size)
        grid = (num_blocks,)
        
        compute_identity_kernel[grid](
            top_k=top_k,
            hidden_states_ptr=hidden_states,
            expert_scales_ptr=expert_scales,
            num_tokens=num_tokens,
            output_ptr=output,
            hidden_dim=hidden_dim,
            scales_stride=expert_scales.stride(0),
            BLOCK_SIZE=self.block_size,
        )
        
        return output


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    dtype = torch.float16
    
    num_tokens = 64
    hidden_dim = 512
    top_k = 8
    
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    expert_scales = torch.randn(num_tokens, top_k, dtype=torch.float32, device=device).abs()
    # 归一化scales
    expert_scales = expert_scales / expert_scales.sum(dim=-1, keepdim=True)
    
    return [hidden_states, expert_scales]


def get_init_inputs():
    """生成初始化参数"""
    return [128]

