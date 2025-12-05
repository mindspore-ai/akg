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
# 源文件: vllm/v1/worker/gpu/block_table.py
# vLLM函数: BlockTables.append_block_ids
# 功能: 向块表追加新的块ID
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    """加载指针的辅助函数"""
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _append_block_ids_kernel_impl(
    # Inputs
    req_indices,  # [num_reqs]
    cu_num_new_blocks_ptr,  # [num_kv_cache_groups, num_reqs + 1]
    cu_num_new_blocks_stride,
    new_block_ids_ptr,  # [num_kv_cache_groups, num_new_blocks]
    new_block_ids_stride,
    overwrite,  # [num_reqs]
    block_table_strides,  # [num_kv_cache_groups]
    # Outputs
    block_table_ptrs,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    """
    向块表追加新的块ID kernel
    
    将新分配的块ID追加到请求的块表中。
    """
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    req_idx = tl.load(req_indices + batch_idx)
    do_overwrite = tl.load(overwrite + batch_idx)

    group_new_blocks_ptr = cu_num_new_blocks_ptr + group_id * cu_num_new_blocks_stride
    start_idx = tl.load(group_new_blocks_ptr + batch_idx)
    end_idx = tl.load(group_new_blocks_ptr + batch_idx + 1)
    num_new_blocks = end_idx - start_idx

    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    dst_start_idx = tl.load(group_num_blocks_ptr + req_idx) if not do_overwrite else 0
    dst_end_idx = dst_start_idx + num_new_blocks
    tl.store(group_num_blocks_ptr + req_idx, dst_end_idx)

    # Destination
    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    row_ptr = block_table_ptr + req_idx * block_table_stride

    group_new_block_ids_ptr = new_block_ids_ptr + group_id * new_block_ids_stride
    for i in range(0, num_new_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(
            group_new_block_ids_ptr + start_idx + offset, mask=offset < num_new_blocks
        )
        tl.store(
            row_ptr + dst_start_idx + offset, block_ids, mask=offset < num_new_blocks
        )


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, num_kv_cache_groups: int = 1, block_size: int = 1024):
        super().__init__()
        self.num_kv_cache_groups = num_kv_cache_groups
        self.block_size = block_size
        self.block_tables_list = []  # 存储block_tables的引用

    def forward(
        self,
        req_indices: torch.Tensor,
        cu_num_new_blocks: torch.Tensor,
        new_block_ids: torch.Tensor,
        overwrite: torch.Tensor,
        block_table_strides: torch.Tensor,
        block_table_ptrs: torch.Tensor,
        num_blocks: torch.Tensor,
        block_tables: torch.Tensor,  # 新增：实际的block_tables张量
    ) -> torch.Tensor:
        """
        向块表追加新的块ID
        
        Args:
            req_indices: [num_reqs] - 请求索引
            cu_num_new_blocks: [num_kv_cache_groups, num_reqs + 1] - 累积新块数
            new_block_ids: [num_kv_cache_groups, num_new_blocks] - 新块ID
            overwrite: [num_reqs] - 是否覆盖
            block_table_strides: [num_kv_cache_groups] - 块表步长
            block_table_ptrs: [num_kv_cache_groups] - 块表指针
            num_blocks: [num_kv_cache_groups, max_num_reqs] - 块数量
            block_tables: [max_num_reqs, max_num_blocks] - 块表张量
            
        Returns:
            block_tables: 更新后的块表（原地修改）
        """
        num_reqs = req_indices.shape[0]
        
        grid = (self.num_kv_cache_groups, num_reqs)
        
        _append_block_ids_kernel_impl[grid](
            req_indices,
            cu_num_new_blocks,
            cu_num_new_blocks.stride(0),
            new_block_ids,
            new_block_ids.stride(0),
            overwrite,
            block_table_strides,
            block_table_ptrs,
            num_blocks,
            num_blocks.stride(0),
            BLOCK_SIZE=self.block_size,
        )
        
        return block_tables


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, num_kv_cache_groups: int = 1, block_size: int = 1024):
        super().__init__()
        self.num_kv_cache_groups = num_kv_cache_groups
        self.block_size = block_size

    def forward(
        self,
        req_indices: torch.Tensor,
        cu_num_new_blocks: torch.Tensor,
        new_block_ids: torch.Tensor,
        overwrite: torch.Tensor,
        block_table_strides: torch.Tensor,
        block_table_ptrs: torch.Tensor,
        num_blocks: torch.Tensor,
        block_tables: torch.Tensor,  # 新增：实际的block_tables张量
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.block_table import _append_block_ids_kernel

        num_reqs = req_indices.shape[0]
        
        grid = (self.num_kv_cache_groups, num_reqs)
        
        _append_block_ids_kernel[grid](
            req_indices,
            cu_num_new_blocks,
            cu_num_new_blocks.stride(0),
            new_block_ids,
            new_block_ids.stride(0),
            overwrite,
            block_table_strides,
            block_table_ptrs,
            num_blocks,
            num_blocks.stride(0),
            BLOCK_SIZE=self.block_size,
        )
        
        return block_tables


def get_inputs():
    """生成测试输入"""
    
    num_kv_cache_groups = 1
    num_reqs = 4
    max_num_reqs = 16
    max_num_blocks = 32
    num_new_blocks = 8
    
    req_indices = torch.arange(num_reqs, dtype=torch.int32)
    cu_num_new_blocks = torch.tensor([[0, 2, 4, 6, 8]], dtype=torch.int32)
    new_block_ids = torch.randint(0, 1000, (num_kv_cache_groups, num_new_blocks), dtype=torch.int32)
    overwrite = torch.tensor([False, False, False, False], dtype=torch.bool)
    
    # 创建块表
    block_tables = torch.zeros(max_num_reqs, max_num_blocks, dtype=torch.int32)
    block_table_ptrs = torch.tensor([block_tables.data_ptr()], dtype=torch.uint64)
    block_table_strides = torch.tensor([block_tables.stride(0)], dtype=torch.int64)
    num_blocks = torch.zeros(num_kv_cache_groups, max_num_reqs, dtype=torch.int32)
    
    # 返回block_tables作为最后一个参数
    return [req_indices, cu_num_new_blocks, new_block_ids, overwrite, block_table_strides, block_table_ptrs, num_blocks, block_tables]


def get_init_inputs():
    """生成初始化参数"""
    return [1, 1024]

