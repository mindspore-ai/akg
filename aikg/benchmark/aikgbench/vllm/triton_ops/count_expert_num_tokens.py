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
from typing import Optional

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/fused_moe/utils.py
# vLLM函数: count_expert_num_tokens
# 功能: 统计分配给每个专家的token数量（用于MoE）
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _count_expert_num_tokens_impl(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count the number of tokens assigned to each expert in MoE.
    
    统计分配给每个专家(expert)的token数量的Triton kernel。
    """
    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
        if HAS_EXPERT_MAP:
            expert_map_ptrs = expert_map + expert_ids
            expert_map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptrs, mask=expert_map_mask, other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def count_expert_num_tokens_impl(
    topk_ids: torch.Tensor, 
    num_local_experts: int, 
    expert_map: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Count the number of tokens assigned to each expert.

    Parameters:
    - topk_ids (torch.Tensor): Tensor mapping each token to its list of experts.
    - num_local_experts (int): Number of experts in this rank.
    - expert_map (Optional[torch.Tensor]): A tensor mapping expert indices
      from the global expert space to the local expert space.

    Returns:
    A tensor of size num_local_experts, where tensor[i] holds the number
    of tokens assigned to the ith expert.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens_impl[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_num_tokens


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        topk_ids: torch.Tensor,
        num_local_experts: int,
        expert_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        统计分配给每个专家的token数量
        
        Args:
            topk_ids: [num_tokens * topk] - token到专家的映射
            num_local_experts: 专家数量
            expert_map: 可选的专家映射表
            
        Returns:
            expert_num_tokens: [num_local_experts] - 每个专家的token数
        """
        return count_expert_num_tokens_impl(topk_ids, num_local_experts, expert_map)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        topk_ids: torch.Tensor,
        num_local_experts: int,
        expert_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens

        return count_expert_num_tokens(topk_ids, num_local_experts, expert_map)


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    dtype = torch.int64
    
    # MoE场景：8个专家，128个token，每个token选择top-2专家
    num_tokens = 128
    topk = 2
    num_experts = 8
    
    # topk_ids: [num_tokens * topk] - 每个token的top-k专家ID
    # 随机分配专家，范围0-7
    topk_ids = torch.randint(0, num_experts, (num_tokens * topk,), dtype=dtype, device=device)
    
    # 不使用expert_map（简单场景）
    num_local_experts = num_experts
    expert_map = None
    
    return [topk_ids, num_local_experts, expert_map]


def get_init_inputs():
    """生成初始化参数"""
    return []

