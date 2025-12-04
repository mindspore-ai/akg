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
# 源文件: vllm/v1/worker/gpu/input_batch.py
# vLLM函数: combine_sampled_and_draft_tokens
# 功能: 合并采样token和草稿token
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _combine_sampled_and_draft_tokens_kernel_impl(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    合并采样token和草稿token kernel
    
    将最后采样的token和草稿token合并到input_ids，并计算logits索引。
    """
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - 1

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Write the last sampled token ID to input_ids.
    last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
    tl.store(input_ids_ptr + query_end - num_logits, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, num_speculative_steps: int = 5):
        super().__init__()
        self.num_speculative_steps = num_speculative_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_logits: int,
    ) -> torch.Tensor:
        """
        合并采样token和草稿token
        
        Args:
            input_ids: [num_tokens] - 输入token ID（输出）
            idx_mapping: [num_reqs] - batch到请求状态的映射
            last_sampled_tokens: [max_num_reqs] - 最后采样的token
            query_start_loc: [num_reqs + 1] - 累积query位置
            seq_lens: [num_reqs] - 序列长度
            prefill_len: [max_num_reqs] - prefill长度
            draft_tokens: [max_num_reqs, num_speculative_steps] - 草稿tokens
            cu_num_logits: [num_reqs + 1] - 累积logit数
            num_logits: 总logit数
            
        Returns:
            logits_indices: [num_logits] - logits索引
        """
        num_reqs = seq_lens.shape[0]
        
        logits_indices = torch.empty(
            num_logits,
            dtype=torch.int64,
            device=input_ids.device,
        )
        
        _combine_sampled_and_draft_tokens_kernel_impl[(num_reqs,)](
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu_num_logits,
            logits_indices,
            # NOTE(woosuk): Add 1 to ensure the block can cover the last sampled token
            # in addition to all draft tokens.
            BLOCK_SIZE=triton.next_power_of_2(self.num_speculative_steps + 1),
        )
        
        return logits_indices


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, num_speculative_steps: int = 5):
        super().__init__()
        self.num_speculative_steps = num_speculative_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_logits: int,
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.input_batch import (
            combine_sampled_and_draft_tokens,
        )

        return combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            cu_num_logits,
            num_logits,
        )


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    
    num_reqs = 4
    max_num_reqs = 16
    num_tokens = 64
    num_speculative_steps = 5
    num_logits = num_reqs * (num_speculative_steps + 1)
    
    input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    last_sampled_tokens = torch.randint(0, 32000, (max_num_reqs,), dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32, device=device)
    seq_lens = torch.randint(60, 100, (num_reqs,), dtype=torch.int32, device=device)
    prefill_len = torch.randint(50, 80, (max_num_reqs,), dtype=torch.int32, device=device)
    draft_tokens = torch.randint(0, 32000, (max_num_reqs, num_speculative_steps), dtype=torch.int32, device=device)
    cu_num_logits = torch.arange(0, num_logits + 1, num_speculative_steps + 1, dtype=torch.int32, device=device)
    
    return [input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len, draft_tokens, cu_num_logits, num_logits]


def get_init_inputs():
    """生成初始化参数"""
    return [5]

