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
# vLLM函数: compute_token_logprobs
# 功能: Top-K Log Softmax计算
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _topk_log_softmax_kernel_impl(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    """
    Top-K Log Softmax kernel
    
    计算指定token ID的log softmax值。
    """
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)

    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


def compute_token_logprobs_impl(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """计算指定token的logprobs"""
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = torch.empty(
        batch_size,
        num_logprobs,
        dtype=torch.float32,
        device=logits.device,
    )
    _topk_log_softmax_kernel_impl[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        PADDED_TOPK=triton.next_power_of_2(num_logprobs),
    )
    return logprobs


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算Top-K token的log softmax
        
        Args:
            logits: [batch_size, vocab_size]
            token_ids: [batch_size, num_logprobs] - 要计算的token ID
            
        Returns:
            logprobs: [batch_size, num_logprobs]
        """
        return compute_token_logprobs_impl(logits, token_ids)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

        return compute_token_logprobs(logits, token_ids)


def get_inputs():
    """生成测试输入"""
    batch_size = 8
    vocab_size = 32000
    num_logprobs = 10
    
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    token_ids = torch.randint(0, vocab_size, (batch_size, num_logprobs), dtype=torch.long)
    
    return [logits, token_ids]


def get_init_inputs():
    """生成初始化参数"""
    return [1024]

