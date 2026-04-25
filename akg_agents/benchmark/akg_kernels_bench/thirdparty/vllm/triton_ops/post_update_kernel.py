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
from typing import Tuple

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/v1/worker/gpu/input_batch.py
# vLLM函数: post_update
# 功能: 采样后更新状态
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _post_update_kernel_impl(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
):
    """
    采样后更新状态kernel
    
    更新已计算token数、最后采样的token和输出bin计数。
    """
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)

    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        token_ptr = (
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + token_id
        )
        count = tl.load(token_ptr)
        count += 1
        tl.store(token_ptr, count)

    query_start = tl.load(query_start_loc_ptr + req_id)
    query_end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    num_computed += query_len - num_rejected
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        output_bin_counts: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样后更新状态
        
        Args:
            idx_mapping: [num_reqs] - batch到请求状态的映射
            num_computed_tokens: [max_num_reqs] - 已计算token数（输出）
            last_sampled_tokens: [max_num_reqs] - 最后采样的token（输出）
            output_bin_counts: [max_num_reqs, vocab_size] - 输出bin计数（输出）
            sampled_tokens: [num_reqs, num_speculative_steps + 1] - 采样的tokens
            num_sampled: [num_reqs] - 采样的token数
            num_rejected: [num_reqs] - 拒绝的token数
            query_start_loc: [num_reqs + 1] - 累积query位置
            
        Returns:
            (num_computed_tokens, last_sampled_tokens, output_bin_counts): 修改后的三个张量
        """
        num_reqs = idx_mapping.shape[0]
        
        _post_update_kernel_impl[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            output_bin_counts.stride(0),
            sampled_tokens,
            sampled_tokens.stride(0),
            num_sampled,
            num_rejected,
            query_start_loc,
            num_warps=1,
        )
        
        return num_computed_tokens, last_sampled_tokens, output_bin_counts


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        output_bin_counts: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm.v1.worker.gpu.input_batch import post_update

        post_update(
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )
        
        return num_computed_tokens, last_sampled_tokens, output_bin_counts


def get_inputs():
    """生成测试输入"""
    
    num_reqs = 4
    max_num_reqs = 16
    vocab_size = 32000
    num_speculative_steps = 5
    
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    num_computed_tokens = torch.randint(0, 50, (max_num_reqs,), dtype=torch.int32)
    last_sampled_tokens = torch.randint(0, vocab_size, (max_num_reqs,), dtype=torch.int32)
    output_bin_counts = torch.zeros(max_num_reqs, vocab_size, dtype=torch.int32)
    sampled_tokens = torch.randint(0, vocab_size, (num_reqs, num_speculative_steps + 1), dtype=torch.int32)
    num_sampled = torch.randint(1, num_speculative_steps + 1, (num_reqs,), dtype=torch.int32)
    num_rejected = torch.randint(0, 3, (num_reqs,), dtype=torch.int32)
    query_start_loc = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32)
    
    return [idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts, sampled_tokens, num_sampled, num_rejected, query_start_loc]


def get_init_inputs():
    """生成初始化参数"""
    return []

