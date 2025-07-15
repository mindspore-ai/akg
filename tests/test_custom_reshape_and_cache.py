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
""" tests_custom_pyboost_ascend """

import os
import numpy as np
import mindspore as ms
from mindspore.ops import CustomOpBuilder, ModuleWrapper
from mindspore import Tensor, context, Parameter, ops
from tests.mark_utils import arg_mark
import pytest
from vllm_mindspore import ms_custom_ops

@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('np_dtype', [np.float16])
@pytest.mark.parametrize('kv_dim', [3])
def test_custom_reshape_and_cache(exec_mode, np_dtype, kv_dim):
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    
    class MyNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            if exec_mode is context.PYNATIVE_MODE:
                self.reshape_and_cache_func = ms_custom_ops.reshape_and_cache
            else:
                def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, head_num):
                    mod = ModuleWrapper("custom_reshape_and_cache", ms_custom_ops)
                    return mod.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, head_num)
                self.reshape_and_cache_func = reshape_and_cache

        def construct(self, key, value, key_cache, value_cache, slot_mapping, head_num):
            return self.reshape_and_cache_func(key, value, key_cache, value_cache, slot_mapping, head_num)
        
    num_slots = 50
    slot_size = 128
    b = 13
    s = 32
    n = 40
    d = 128
    
    def create_nd_inputs(dtype=np.float16, kv_dim=3):
        cache_shape = (num_slots, slot_size, n, d)
        if kv_dim == 2:
            update_shape = (b * s, n * d)
            num_tokens = update_shape[0]
        elif kv_dim == 3:
            update_shape = (b, s, n * d)
            num_tokens = update_shape[0] * update_shape[1]
        else:
            raise Exception(
                "Key's dim should be 2 or 3, but got {0}".format(kv_dim))

        if dtype == np.int8:
            key_update = np.random.randint(low=-128, high=127,
                                        size=update_shape,
                                        dtype=np.int8)
            value_update = np.random.randint(low=-128, high=127,
                                            size=update_shape,
                                            dtype=np.int8)
            key_cache = np.random.randint(low=-128, high=127,
                                        size=cache_shape,
                                        dtype=np.int8)
            value_cache = np.random.randint(low=-128, high=127,
                                            size=cache_shape,
                                            dtype=np.int8)
        else:
            key_update = np.random.rand(*update_shape).astype(dtype)
            value_update = np.random.rand(*update_shape).astype(dtype)
            key_cache = np.random.rand(*cache_shape).astype(dtype)
            value_cache = np.random.rand(*cache_shape).astype(dtype)

        slot_map = np.random.choice(np.arange(num_tokens), num_tokens,
                                    replace=False).astype(np.int32)

        return key_update, value_update, key_cache, value_cache, slot_map
    
    def nd_inference(key, value, key_cache, value_cache, slot_map):
        key_tmp = key.copy()
        value_tmp = value.copy()
        key_cache_ans = key_cache.copy()
        value_cache_ans = value_cache.copy()
        head = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        key_tmp = key_tmp.reshape(-1, head, head_dim)
        value_tmp = value_tmp.reshape(-1, head, head_dim)
        for i, slot in enumerate(slot_map):
            slot_idx = slot // key_cache.shape[1]
            slot_offset = slot % key_cache.shape[1]
            key_cache_ans[slot_idx][slot_offset] = key_tmp[i]
            value_cache_ans[slot_idx][slot_offset] = value_tmp[i]
        return key_cache_ans, value_cache_ans
    
    def create_ms_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map, format="", exec_mode=context.GRAPH_MODE):
        ms_key = Tensor(np_k)
        ms_value = Tensor(np_v)
        if exec_mode == context.GRAPH_MODE:
            ms_key_cache = Parameter(Tensor(np_k_cache), storage_format=format, name="key_cache")
            ms_value_cache = Parameter(Tensor(np_v_cache), storage_format=format, name="value_cache")
        else:
            ms_key_cache = Tensor(np_k_cache)
            ms_value_cache = Tensor(np_v_cache)
        ms_slot_map = Tensor(np_slot_map)
        return ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map
    
    np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nd_inputs(
        np_dtype, kv_dim)
    np_k_cache_out, np_v_cache_out = nd_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
    _ = MyNet()(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, n)

    # if np_dtype == bfloat16:
    #     assert np.allclose(ms_k_cache.float().asnumpy(),
    #                        np_k_cache_out.astype(np.float32), 0.001, 0.001)
    #     assert np.allclose(ms_v_cache.float().asnumpy(),
    #                        np_v_cache_out.astype(np.float32), 0.001, 0.001)
    # else:
    #     assert np.allclose(ms_k_cache.asnumpy(), np_k_cache_out, 0.001, 0.001)
    #     assert np.allclose(ms_v_cache.asnumpy(), np_v_cache_out, 0.001, 0.001)

    assert np.allclose(ms_k_cache.asnumpy(), np_k_cache_out, 0.001, 0.001)
    assert np.allclose(ms_v_cache.asnumpy(), np_v_cache_out, 0.001, 0.001)

# test_custom_reshape_and_cache(context.PYNATIVE_MODE, np.float16, 3)
