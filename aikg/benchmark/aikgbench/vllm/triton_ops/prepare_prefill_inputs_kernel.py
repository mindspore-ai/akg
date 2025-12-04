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
# vLLM函数: prepare_prefill_inputs
# 功能: 准备prefill输入数据
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _prepare_prefill_inputs_kernel_impl(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_token_ids_ptr,
    prefill_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    准备prefill输入kernel
    
    将prefill token复制到input_ids，并准备下一个prefill token。
    """
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    prefill_ptr = prefill_token_ids_ptr + req_state_idx * prefill_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(prefill_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    next_pos = num_computed + query_len
    if next_pos < prefill_len:
        next_token = tl.load(prefill_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_token_ids: torch.Tensor,
        prefill_len: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备prefill输入
        
        Args:
            input_ids: [num_tokens] - 输入token ID（输出）
            next_prefill_tokens: [max_num_reqs] - 下一个prefill token（输出）
            idx_mapping: [num_reqs] - batch到请求状态的映射
            query_start_loc: [num_reqs + 1] - 累积query位置
            prefill_token_ids: [max_num_reqs, max_prefill_len] - prefill tokens
            prefill_len: [max_num_reqs] - prefill长度
            num_computed_tokens: [max_num_reqs] - 已计算token数
            
        Returns:
            (input_ids, next_prefill_tokens): 修改后的input_ids和next_prefill_tokens
        """
        num_reqs = idx_mapping.shape[0]
        
        _prepare_prefill_inputs_kernel_impl[(num_reqs,)](
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            prefill_token_ids,
            prefill_token_ids.stride(0),
            prefill_len,
            num_computed_tokens,
            BLOCK_SIZE=self.block_size,
        )
        
        return input_ids, next_prefill_tokens


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_token_ids: torch.Tensor,
        prefill_len: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm.v1.worker.gpu.input_batch import prepare_prefill_inputs

        prepare_prefill_inputs(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            prefill_token_ids,
            prefill_len,
            num_computed_tokens,
        )
        
        return input_ids, next_prefill_tokens


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    
    num_reqs = 4
    max_num_reqs = 16
    num_tokens = 64
    max_prefill_len = 128
    
    input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32, device=device)
    prefill_token_ids = torch.randint(0, 32000, (max_num_reqs, max_prefill_len), dtype=torch.int32, device=device)
    prefill_len = torch.randint(50, 100, (max_num_reqs,), dtype=torch.int32, device=device)
    num_computed_tokens = torch.randint(0, 20, (max_num_reqs,), dtype=torch.int32, device=device)
    
    return [input_ids, next_prefill_tokens, idx_mapping, query_start_loc, prefill_token_ids, prefill_len, num_computed_tokens]


def get_init_inputs():
    """生成初始化参数"""
    return [1024]

