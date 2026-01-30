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
# vLLM函数: BlockTables.gather_block_tables
# 功能: 根据索引收集块表
# 测试文件: 无专门测试文件
# ============================================================================


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    """加载指针的辅助函数"""
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _gather_block_tables_kernel_impl(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    收集块表kernel
    
    根据batch索引到请求索引的映射，收集对应的块表。
    """
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)

    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    stride = tl.load(block_table_strides + group_id)
    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


class Model(nn.Module):
    """原生PyTorch实现（直接调用复制的Triton kernel）"""

    def __init__(self, num_kv_cache_groups: int = 1, block_size: int = 1024):
        super().__init__()
        self.num_kv_cache_groups = num_kv_cache_groups
        self.block_size = block_size

    def forward(
        self,
        idx_mapping: torch.Tensor,
        src_block_table_ptrs: torch.Tensor,
        dst_block_table_ptrs: torch.Tensor,
        block_table_strides: torch.Tensor,
        num_blocks: torch.Tensor,
        dst_block_tables: torch.Tensor,  # 新增：实际的目标块表张量
    ) -> torch.Tensor:
        """
        根据索引映射收集块表
        
        Args:
            idx_mapping: [num_reqs] - batch索引到请求索引的映射
            src_block_table_ptrs: [num_kv_cache_groups] - 源块表指针
            dst_block_table_ptrs: [num_kv_cache_groups] - 目标块表指针
            block_table_strides: [num_kv_cache_groups] - 块表步长
            num_blocks: [num_kv_cache_groups, max_num_reqs] - 块数量
            dst_block_tables: [max_num_reqs, max_num_blocks] - 目标块表张量
            
        Returns:
            dst_block_tables: 更新后的目标块表（原地修改）
        """
        num_reqs = idx_mapping.shape[0]
        
        grid = (self.num_kv_cache_groups, num_reqs)
        
        _gather_block_tables_kernel_impl[grid](
            idx_mapping,
            src_block_table_ptrs,
            dst_block_table_ptrs,
            block_table_strides,
            num_blocks,
            num_blocks.stride(0),
            BLOCK_SIZE=self.block_size,
        )
        
        return dst_block_tables


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""

    def __init__(self, num_kv_cache_groups: int = 1, block_size: int = 1024):
        super().__init__()
        self.num_kv_cache_groups = num_kv_cache_groups
        self.block_size = block_size

    def forward(
        self,
        idx_mapping: torch.Tensor,
        src_block_table_ptrs: torch.Tensor,
        dst_block_table_ptrs: torch.Tensor,
        block_table_strides: torch.Tensor,
        num_blocks: torch.Tensor,
        dst_block_tables: torch.Tensor,  # 新增：实际的目标块表张量
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

        num_reqs = idx_mapping.shape[0]
        
        grid = (self.num_kv_cache_groups, num_reqs)
        
        _gather_block_tables_kernel[grid](
            idx_mapping,
            src_block_table_ptrs,
            dst_block_table_ptrs,
            block_table_strides,
            num_blocks,
            num_blocks.stride(0),
            BLOCK_SIZE=self.block_size,
        )
        
        return dst_block_tables


def get_inputs():
    """生成测试输入"""
    
    num_kv_cache_groups = 1
    num_reqs = 4
    max_num_reqs = 16
    max_num_blocks = 32
    
    idx_mapping = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
    
    # 创建源和目标块表
    src_block_tables = torch.randint(0, 1000, (max_num_reqs, max_num_blocks), dtype=torch.int32)
    dst_block_tables = torch.zeros(max_num_reqs, max_num_blocks, dtype=torch.int32)
    
    src_block_table_ptrs = torch.tensor([src_block_tables.data_ptr()], dtype=torch.uint64)
    dst_block_table_ptrs = torch.tensor([dst_block_tables.data_ptr()], dtype=torch.uint64)
    block_table_strides = torch.tensor([src_block_tables.stride(0)], dtype=torch.int64)
    num_blocks = torch.randint(1, 16, (num_kv_cache_groups, max_num_reqs), dtype=torch.int32)
    
    # 返回dst_block_tables作为最后一个参数
    return [idx_mapping, src_block_table_ptrs, dst_block_table_ptrs, block_table_strides, num_blocks, dst_block_tables]


def get_init_inputs():
    """生成初始化参数"""
    return [1, 1024]

