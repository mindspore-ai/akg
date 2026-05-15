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
# vLLM函数: prepare_pos_seq_lens
# 功能: 准备位置和序列长度
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _prepare_pos_seq_lens_kernel_impl(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    """
    准备位置和序列长度kernel
    
    计算每个token的位置和每个请求的序列长度。
    """
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备位置和序列长度
        
        Args:
            idx_mapping: [num_reqs] - batch到请求状态的映射
            query_start_loc: [num_reqs + 1] - 累积query位置
            num_computed_tokens: [max_num_reqs] - 已计算token数
            pos: [num_tokens] - token位置（输出）
            seq_lens: [max_num_reqs] - 序列长度（输出）
            
        Returns:
            (pos, seq_lens): 修改后的pos和seq_lens
        """
        num_reqs = idx_mapping.shape[0]
        max_num_reqs = seq_lens.shape[0]
        
        # NOTE(woosuk): We do +1 because the last thread block is used
        # to pad unused seq_lens as 0 for full CUDA graphs.
        _prepare_pos_seq_lens_kernel_impl[(num_reqs + 1,)](
            pos,
            seq_lens,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            max_num_reqs,
            BLOCK_SIZE=self.block_size,
        )
        
        return pos, seq_lens


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens

        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            pos,
            seq_lens,
        )
        
        return pos, seq_lens


def get_inputs():
    """生成测试输入"""
    
    num_reqs = 4
    max_num_reqs = 16
    num_tokens = 64
    
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32)
    num_computed_tokens = torch.randint(0, 50, (max_num_reqs,), dtype=torch.int32)
    pos = torch.zeros(num_tokens, dtype=torch.int64)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    
    return [idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens]


def get_init_inputs():
    """生成初始化参数"""
    return [1024]

