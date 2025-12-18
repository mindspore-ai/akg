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
# 源文件: vllm/v1/worker/gpu/sample/logprob.py
# vLLM函数: compute_topk_logprobs (内部调用)
# 功能: 计算token在logits中的排名
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _ranks_kernel_impl(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    计算token排名kernel
    
    计算给定token在logits中的排名（有多少token的logit大于它）。
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits > x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_ranks_impl(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    block_size: int = 8192,
) -> torch.Tensor:
    """计算token排名的Python包装函数"""
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]
    token_ranks = torch.empty(
        batch_size,
        dtype=torch.int64,
        device=logits.device,
    )
    _ranks_kernel_impl[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        token_ids,
        vocab_size,
        BLOCK_SIZE=block_size,  # type: ignore
    )
    return token_ranks


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, block_size: int = 8192):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算token排名
        
        Args:
            logits: [batch_size, vocab_size]
            token_ids: [batch_size] - 要查询排名的token ID
            
        Returns:
            ranks: [batch_size] - 每个token的排名
        """
        return compute_ranks_impl(logits, token_ids, self.block_size)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, block_size: int = 8192):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # vLLM没有直接导出ranks_kernel，我们直接调用它
        from vllm.v1.worker.gpu.sample.logprob import _ranks_kernel

        batch_size = logits.shape[0]
        vocab_size = logits.shape[1]
        token_ranks = torch.empty(
            batch_size,
            dtype=torch.int64,
            device=logits.device,
        )
        _ranks_kernel[(batch_size,)](
            token_ranks,
            logits,
            logits.stride(0),
            token_ids,
            vocab_size,
            BLOCK_SIZE=self.block_size,  # type: ignore
        )
        return token_ranks


def get_inputs():
    """生成测试输入"""
    
    batch_size = 8
    vocab_size = 32000
    
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    token_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long)
    
    return [logits, token_ids]


def get_init_inputs():
    """生成初始化参数"""
    return [8192]

