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
# 源文件: vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py
# vLLM函数: pack_bitmatrix (kernel only)
# 功能: 将topk_ids打包成位矩阵（用于MoE路由）
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def pack_bitmatrix_kernel(
    bitmatrix,
    topk_ids,
    n_rows,  # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Packs topk_ids into a bitmatrix.
    
    将topk_ids打包成位矩阵的Triton kernel，用于MoE路由。
    
    参考: https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/distributed.py
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    # Iterate through all the relevant bitmatrix columns.
    for i in range(bm_cols):
        # When BLOCK_SIZE_K=32, offs is just the column index.
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # All topks that need to go into this column has the correct bit set.
        # Other bits are 0. x is a 2D tensor.
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        # Reduce x to get a single int32_t bitpack.
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def pack_bitmatrix_impl(
    topk_ids: torch.Tensor,
    n_rows: int,
    n_experts: int,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_K: int = 32,
) -> torch.Tensor:
    """
    Pack topk_ids into a bitmatrix format.
    
    Args:
        topk_ids: [n_rows, topk] - indices of top-k experts for each row
        n_rows: number of rows
        n_experts: total number of experts
        BLOCK_SIZE_M: block size for M dimension
        BLOCK_SIZE_K: block size for K dimension
    
    Returns:
        bitmatrix: [n_rows, bm_cols] - packed bitmatrix (uint32)
    """
    n_expts_act = topk_ids.shape[1]  # topk
    bm_cols = triton.cdiv(n_experts, 32)  # number of uint32 columns needed
    
    # Create output bitmatrix
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )
    
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    
    pack_bitmatrix_kernel[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        n_expts_act,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return bitmatrix


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, block_size_m: int = 32, block_size_k: int = 32):
        super().__init__()
        self.block_size_m = block_size_m
        self.block_size_k = block_size_k

    def forward(
        self,
        topk_ids: torch.Tensor,
        n_experts: int,
    ) -> torch.Tensor:
        """
        将topk_ids打包成位矩阵
        
        Args:
            topk_ids: [n_rows, topk] - 每行的top-k专家索引
            n_experts: 专家总数
            
        Returns:
            bitmatrix: [n_rows, bm_cols] - 打包的位矩阵（转换为int32以便验证）
        """
        n_rows = topk_ids.shape[0]
        bitmatrix = pack_bitmatrix_impl(
            topk_ids, n_rows, n_experts, self.block_size_m, self.block_size_k
        )
        # 转换为int32以便验证（uint32不支持索引操作）
        return bitmatrix.to(torch.int32)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, block_size_m: int = 32, block_size_k: int = 32):
        super().__init__()
        self.block_size_m = block_size_m
        self.block_size_k = block_size_k

    def forward(
        self,
        topk_ids: torch.Tensor,
        n_experts: int,
    ) -> torch.Tensor:
        # vLLM中pack_bitmatrix是@triton.jit kernel，需要手动调用
        # 这里我们复制相同的实现逻辑
        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
            pack_bitmatrix,
        )

        n_rows = topk_ids.shape[0]
        n_expts_act = topk_ids.shape[1]
        bm_cols = triton.cdiv(n_experts, 32)

        bitmatrix = torch.zeros(
            (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
        )

        grid = (triton.cdiv(n_rows, self.block_size_m),)

        pack_bitmatrix[grid](
            bitmatrix,
            topk_ids,
            n_rows,
            bm_cols,
            n_expts_act,
            BLOCK_SIZE_M=self.block_size_m,
            BLOCK_SIZE_K=self.block_size_k,
        )

        # 转换为int32以便验证（uint32不支持索引操作）
        return bitmatrix.to(torch.int32)


def get_inputs():
    """生成测试输入"""
    device = "cuda"
    
    # MoE场景：128行，8个专家，每行选top-2
    n_rows = 128
    topk = 2
    n_experts = 8
    
    # topk_ids: [n_rows, topk] - 每行的top-k专家ID
    topk_ids = torch.randint(0, n_experts, (n_rows, topk), dtype=torch.int64, device=device)
    
    return [topk_ids, n_experts]


def get_init_inputs():
    """生成初始化参数"""
    return [32, 32]  # block_size_m, block_size_k

