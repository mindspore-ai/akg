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
# 源文件: vllm/v1/worker/gpu/sample/gumbel.py
# vLLM函数: gumbel_sample
# 功能: Gumbel采样kernel
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _gumbel_sample_kernel_impl(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    """
    Gumbel采样kernel
    
    对logits应用Gumbel噪声并进行采样。
    """
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + req_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    temp = tl.load(temp_ptr + req_idx).to(tl.float32)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_idx)
        pos = tl.load(pos_ptr + req_idx)
        gumbel_seed = tl.randint(seed, pos)

        # Generate gumbel noise.
        r = tl.rand(gumbel_seed, block).to(tl.float64)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
        gumbel_noise = gumbel_noise.to(tl.float32)

        # Apply temperature.
        if APPLY_TEMPERATURE:
            # NOTE(woosuk): Match the behavior of _penalties_and_temperature_kernel.
            # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
            logits = logits / temp

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    idx = tl.argmax(logits, axis=0)
    token_id = block_idx * BLOCK_SIZE + idx
    value = tl.max(logits, axis=0)
    tl.store(local_argmax_ptr + req_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + req_idx * local_max_stride + block_idx, value)


def gumbel_sample_impl(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
) -> torch.Tensor:
    """Gumbel采样的Python包装函数"""
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_reqs,
        num_blocks,
        dtype=torch.int64,
        device=logits.device,
    )
    local_max = torch.empty(
        num_reqs,
        num_blocks,
        dtype=torch.float32,
        device=logits.device,
    )
    _gumbel_sample_kernel_impl[(num_reqs, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        apply_temperature: bool,
    ) -> torch.Tensor:
        """
        Gumbel采样
        
        Args:
            logits: [num_reqs, vocab_size]
            temperature: [num_reqs]
            seed: [num_reqs]
            pos: [num_reqs]
            apply_temperature: 是否应用温度
            
        Returns:
            sampled: [num_reqs] - 采样的token ID
        """
        return gumbel_sample_impl(logits, temperature, seed, pos, apply_temperature)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        apply_temperature: bool,
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        return gumbel_sample(logits, temperature, seed, pos, apply_temperature)


def get_inputs():
    """生成测试输入"""
    
    num_reqs = 8
    vocab_size = 32000
    
    logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32)
    temperature = torch.ones(num_reqs, dtype=torch.float32)
    seed = torch.randint(0, 2**31, (num_reqs,), dtype=torch.long)
    pos = torch.randint(0, 1000, (num_reqs,), dtype=torch.long)
    apply_temperature = True
    
    return [logits, temperature, seed, pos, apply_temperature]


def get_init_inputs():
    """生成初始化参数"""
    return [1024]

