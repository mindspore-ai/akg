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
# ============================================================================

import numpy as np
import pytest
from mindspore import Tensor, context
import mindspore as ms
import random
import ms_custom_ops
from st_utils import custom_compare

class AsdPagedCacheLoadCustom(ms.nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, key_cache, value_cache, block_table, seq_lens, key, value, seq_starts, kv_cache_cfg,
                  is_seq_lens_cumsum_type, has_seq_starts):
        return ms_custom_ops.paged_cache_load(key_cache, value_cache, block_table, seq_lens, key, value,
                                              seq_starts, kv_cache_cfg, is_seq_lens_cumsum_type,
                                              has_seq_starts)
    
def golden_calc_nd(num_tokens, num_heads, head_size_k, head_size_v, block_size, block_tables, context_lens,
                   seq_starts, key_cache, value_cache, dtype):
    sum_context_lens = context_lens[-1]
    if dtype == ms.float16:
        key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float16)
        value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float16)
    elif dtype == ms.bfloat16:
        key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float32)
        value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float32)
    else:
        key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.int8)
        value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.int8)
    kv_rslt_id = 0
    context_start = 0
    for i in range(num_tokens):
        block_table = block_tables[i]
        context_end = int(context_lens[i + 1])
        context_len = context_end - context_start
        context_start = context_end
        block_table_offset = seq_starts[i] // block_size
        for j in range(context_len):
            block_id = int(block_table[block_table_offset + j // block_size])
            block_offset = j % block_size
            if block_id < 0:
                continue
            temp_k = key_cache[block_id][block_offset]
            temp_v = value_cache[block_id][block_offset]
            key_expect[kv_rslt_id] = temp_k
            value_expect[kv_rslt_id] = temp_v
            kv_rslt_id += 1
    return key_expect, value_expect

def golden_calc_nz(num_tokens, num_heads, head_size_k, head_size_v, block_size, block_tables, context_lens,
                   key_cache, value_cache, dtype):
    sum_context_lens = sum(context_lens)
    if dtype == ms.float16:
        key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
        value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)
    elif dtype == ms.bfloat16:
        key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float32)
        value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float32)
    else:
        key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.int8)
        value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.int8)
    elenum_aligned = 16
    if dtype != ms.float16 and dtype != ms.bfloat16:
        elenum_aligned = 32

    kv_rslt_id = 0    
    for i in range(num_tokens):
        block_table = block_tables[i]
        context_len = int(context_lens[i])
        for j in range(context_len):
            block_id = int(block_table[j // block_size])
            block_offset = j % block_size
            if block_id < 0:
                continue
            temp_k = np.zeros((num_heads * head_size_k))
            temp_v = np.zeros((num_heads * head_size_v))

            for k in range(num_heads * head_size_k // elenum_aligned):
                temp_k[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = (
                    key_cache[block_id][k][block_offset][:]
                )
            for k in range(num_heads * head_size_v // elenum_aligned):
                temp_v[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = (
                    value_cache[block_id][k][block_offset][:]
                )
            key_expect[kv_rslt_id] = temp_k
            value_expect[kv_rslt_id] = temp_v
            kv_rslt_id += 1
    return key_expect, value_expect

def generate_data_nd(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype):
    if dtype == ms.float16:
        key_cache = np.random.randint(1, 11, 
                                      size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float16)
        value_cache = np.random.randint(1, 11, 
                                        size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float16)
    elif dtype == ms.bfloat16:
        key_cache = np.random.randint(1, 11, 
                                      size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
        value_cache = np.random.randint(1, 11, 
                                        size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float32)
    else:
        key_cache = np.random.randint(1, 11, 
                                      size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.int8)
        value_cache = np.random.randint(1, 11, 
                                        size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.int8)
    context_lens = [random.randint(1, 1024) for _ in range(num_tokens)]
    max_context_len = max(context_lens)
    max_num_blocks_per_req = (max_context_len + block_size -1) // block_size + 4
    block_tables = []
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
        ]
        block_tables.append(block_table)
    cu_context_lens = [0]
    for elem in context_lens:
        cu_context_lens.append(cu_context_lens[-1] + elem)
    seq_starts = [random.randint(0, 4) * block_size for _ in range(num_tokens)]
    context_lens = np.array(cu_context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)
    seq_starts = np.array(seq_starts).astype(np.int32)
    sum_context_lens = context_lens[-1]
    key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(key_cache.dtype)
    value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(value_cache.dtype)
    key_tensor = Tensor(key).astype(dtype)
    value_tensor = Tensor(value).astype(dtype)

    return key_cache, value_cache, block_tables, context_lens, key_tensor, value_tensor, seq_starts

def generate_data_nz(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype):
    if dtype == ms.float16:
        key_cache = np.random.randint(
            1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float16)
        value_cache = np.random.randint(
            1, 11, size=(num_blocks, num_heads * head_size_v // 16, block_size, 16)).astype(np.float16)
    elif dtype == ms.bfloat16:
        key_cache = np.random.randint(
            1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float32)
        value_cache = np.random.randint(
            1, 11, size=(num_blocks, num_heads * head_size_v // 16, block_size, 16)).astype(np.float32)
    else:
        key_cache = np.random.randint(
            1, 11, size=(num_blocks, num_heads * head_size_k // 32, block_size, 32)).astype(np.int8)
        value_cache = np.random.randint(
            1, 11, size=(num_blocks, num_heads * head_size_v // 32, block_size, 32)).astype(np.int8)
    context_lens = [random.randint(1, 1024) for _ in range(num_tokens)]
    max_context_len = max(context_lens)
    max_num_blocks_per_req = (max_context_len + block_size -1) // block_size
    block_tables = []
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
        ]
        block_tables.append(block_table)

    context_lens = np.array(context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)
    sum_context_lens = sum(context_lens)
    key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(key_cache.dtype)
    value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(value_cache.dtype)
    key_tensor = Tensor(key).astype(dtype)
    value_tensor = Tensor(value).astype(dtype)

    return key_cache, value_cache, block_tables, context_lens, key_tensor, value_tensor, None

def paged_cache_load_function(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype,
                              format_type, cu_seq_lens, has_seq_starts):
    if format_type == 0:
        key_cache, value_cache, block_tables, context_lens, key_tensor, value_tensor, seq_starts = (
            generate_data_nd(
                num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype
            )
        )
    else:
        key_cache, value_cache, block_tables, context_lens, key_tensor, value_tensor, seq_starts = (
            generate_data_nz(
                num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype
            )
        )
    seq_starts_tensor = None if seq_starts is None else Tensor(seq_starts)
    net = AsdPagedCacheLoadCustom()
    key_out, value_out = net(
        Tensor(key_cache).astype(dtype),
        Tensor(value_cache).astype(dtype),
        Tensor(block_tables),
        Tensor(context_lens),
        key_tensor,
        value_tensor,
        seq_starts_tensor,
        format_type, cu_seq_lens, has_seq_starts
    )

    if format_type == 0:
        key_golden, value_golden = golden_calc_nd(num_tokens, num_heads, head_size_k, head_size_v, block_size,
                                                  block_tables, context_lens, seq_starts, key_cache, value_cache,
                                                  dtype)
    else:
        key_golden, value_golden = golden_calc_nz(num_tokens, num_heads, head_size_k, head_size_v, block_size,
                                                  block_tables, context_lens, key_cache, value_cache, dtype)
    if dtype == ms.bfloat16:
        key_out_np = key_out.astype(ms.float32).asnumpy()
        value_out_np = value_out.astype(ms.float32).asnumpy()
    else:
        key_out_np = key_out.asnumpy()
        value_out_np = value_out.asnumpy()
    key_out_compare = custom_compare(key_out_np, key_golden, dtype)
    assert key_out_compare, "key_out compare failed"
    value_out_compare = custom_compare(value_out_np, value_golden, dtype)
    assert value_out_compare, "key_out compare failed"

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.float16, ms.int8, ms.bfloat16])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('input_param', [[128, 128, 16, 144, 128, 16, 1],
                                         [256, 64, 16, 192, 128, 32, 1]])
def test_paged_cache_load_nd_with_seq_starts(dtype, context_mode, input_param):
    """
    Feature: test paged_cache_load operator
    Description: test paged_cache_load
    Expectation: the result is correct
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    num_blocks, block_size, num_heads, head_size_k, head_size_v, batch, seq_len = input_param
    num_tokens = batch * seq_len
    dtype = dtype
    format_type = 0 # 0-nd, 1-nz
    cu_seq_lens = True
    has_seq_starts = True
    paged_cache_load_function(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype,
                              format_type, cu_seq_lens, has_seq_starts)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.float16, ms.int8, ms.bfloat16])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('input_param', [[128, 128, 16, 144, 128, 16, 1],
                                         [256, 64, 16, 192, 128, 32, 1]])
def test_paged_cache_load_nz(dtype, context_mode, input_param):
    """
    Feature: test paged_cache_load operator
    Description: test paged_cache_load
    Expectation: the result is correct
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    num_blocks, block_size, num_heads, head_size_k, head_size_v, batch, seq_len = input_param
    num_tokens = batch * seq_len
    dtype = dtype
    format_type = 1 # 0-nd, 1-nz
    cu_seq_lens = False
    has_seq_starts = False
    paged_cache_load_function(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype,
                              format_type, cu_seq_lens, has_seq_starts)
