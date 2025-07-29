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

# Standard library imports
from enum import Enum
from typing import Tuple, Optional, Dict, Any

# Third-party imports
import numpy as np
import pytest

# MindSpore imports
import mindspore as ms
from mindspore import Tensor, context, Parameter, ops, nn
from mindspore.common.api import jit
from mindspore.common.np_dtype import bfloat16

# Local imports
import ms_custom_ops

# Global constants
NUM_SLOTS = 20
SLOT_SIZE = 64
BATCH_SIZE = 13
SEQ_LEN = 3
NUM_HEADS = 16
HEAD_DIM = 32


class CacheFormat(Enum):
    """Cache format enumeration"""
    ND = "nd"
    NZ = "nz"


class DataType(Enum):
    """Data type enumeration"""
    FLOAT16 = np.float16
    BFLOAT16 = bfloat16
    INT8 = np.int8


class ReshapeAndCacheAll(nn.Cell):
    """Reshape and cache operation for NZ/ND format with all parameters"""
    
    def __init__(self):
        super().__init__()

    @jit
    def construct(self, key, value, key_cache, value_cache, slot_map, head_num=0):
        return ms_custom_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slot_map, head_num)


class ReshapeAndCacheKey(nn.Cell):
    """Reshape and cache operation for NZ/ND format with key only"""
    
    def __init__(self):
        super().__init__()

    def construct(self, key, key_cache, slot_map):
        return ms_custom_ops.reshape_and_cache(
            key, key_cache=key_cache, slot_mapping=slot_map)


class MindSporeInputFactory:
    """Factory for creating MindSpore inputs"""
    
    @staticmethod
    def create_inputs(np_k: np.ndarray, np_v: np.ndarray, 
                     np_k_cache: np.ndarray, np_v_cache: np.ndarray, 
                     np_slot_map: np.ndarray, format: str = "", 
                     exec_mode: context = context.GRAPH_MODE) -> Tuple[Tensor, ...]:
        """Create MindSpore inputs"""
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


def create_ms_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map, format="", exec_mode=context.GRAPH_MODE):
    """Legacy function for backward compatibility"""
    return MindSporeInputFactory.create_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map, format, exec_mode)


class TestResultVerifier:
    """Verify test results"""
    
    @staticmethod
    def verify_results(ms_cache: Tensor, np_cache: np.ndarray, 
                      dtype: np.dtype, rtol: float = 0.001, atol: float = 0.001) -> None:
        """Verify results with appropriate dtype handling"""
        if dtype == bfloat16:
            ms_cache_np = ms_cache.float().asnumpy()
            np_cache = np_cache.astype(np.float32)
        else:
            ms_cache_np = ms_cache.asnumpy()
        
        assert np.allclose(ms_cache_np, np_cache, rtol=rtol, atol=atol)


class TestConfig:
    """Test configuration"""
    
    def __init__(self, device_target: str = "Ascend", mode: context = context.GRAPH_MODE,
                 jit_config: Optional[Dict[str, Any]] = None):
        self.device_target = device_target
        self.mode = mode
        self.jit_config = jit_config or {}
    
    def apply(self):
        """Apply test configuration"""
        context.set_context(device_target=self.device_target, mode=self.mode)
        if self.jit_config:
            context.set_context(jit_config=self.jit_config)


# ===============================
#        test nd format
# ===============================
class TestDataGenerator:
    """Data generator for test inputs"""
    
    @staticmethod
    def create_random_data(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Create random data with specified shape and dtype"""
        if dtype == np.int8:
            return np.random.randint(low=-128, high=127, size=shape, dtype=np.int8)
        else:
            return np.random.rand(*shape).astype(dtype)
    
    @staticmethod
    def create_slot_map(num_tokens: int) -> np.ndarray:
        """Create slot mapping"""
        return np.random.choice(np.arange(num_tokens), num_tokens, replace=False).astype(np.int32)
    
    @staticmethod
    def get_update_shape(kv_dim: int) -> Tuple[Tuple[int, ...], int]:
        """Get update shape and number of tokens based on dimension"""
        if kv_dim == 2:
            update_shape = (BATCH_SIZE * SEQ_LEN, NUM_HEADS * HEAD_DIM)
            num_tokens = update_shape[0]
        elif kv_dim == 3:
            update_shape = (BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM)
            num_tokens = update_shape[0] * update_shape[1]
        else:
            raise ValueError(f"Key's dim should be 2 or 3, but got {kv_dim}")
        return update_shape, num_tokens


class NDDataGenerator(TestDataGenerator):
    """Data generator for ND format"""
    
    @staticmethod
    def create_inputs(dtype: np.dtype, kv_dim: int) -> Tuple[np.ndarray, ...]:
        """Create ND format inputs"""
        cache_shape = (NUM_SLOTS, SLOT_SIZE, NUM_HEADS, HEAD_DIM)
        update_shape, num_tokens = TestDataGenerator.get_update_shape(kv_dim)
        
        key_update = TestDataGenerator.create_random_data(update_shape, dtype)
        value_update = TestDataGenerator.create_random_data(update_shape, dtype)
        key_cache = TestDataGenerator.create_random_data(cache_shape, dtype)
        value_cache = TestDataGenerator.create_random_data(cache_shape, dtype)
        slot_map = TestDataGenerator.create_slot_map(num_tokens)
        
        return key_update, value_update, key_cache, value_cache, slot_map


def create_nd_inputs(dtype=np.float16, kv_dim=3):
    """Legacy function for backward compatibility"""
    return NDDataGenerator.create_inputs(dtype, kv_dim)


class InferenceEngine:
    """Inference engine for different formats"""
    
    @staticmethod
    def nd_inference(key: np.ndarray, value: np.ndarray, 
                    key_cache: np.ndarray, value_cache: np.ndarray, 
                    slot_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ND format inference"""
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
    
    @staticmethod
    def nz_inference(key: np.ndarray, value: np.ndarray,
                    key_cache: np.ndarray, value_cache: np.ndarray,
                    slot_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """NZ format inference"""
        key_tmp = key.copy()
        value_tmp = value.copy()
        key_cache_ans = key_cache.copy()
        value_cache_ans = value_cache.copy()
        
        key_tmp = key_tmp.reshape(-1, key_cache.shape[2])
        value_tmp = value_tmp.reshape(-1, key_cache.shape[2])
        
        for i, slot in enumerate(slot_map):
            slot_idx = slot // key_cache.shape[1]
            slot_offset = slot % key_cache.shape[1]
            key_cache_ans[slot_idx][slot_offset] = key_tmp[i]
            value_cache_ans[slot_idx][slot_offset] = value_tmp[i]
        
        return key_cache_ans, value_cache_ans


def nd_inference(key, value, key_cache, value_cache, slot_map):
    """Legacy function for backward compatibility"""
    return InferenceEngine.nd_inference(key, value, key_cache, value_cache, slot_map)


def nz_inference(key, value, key_cache, value_cache, slot_map):
    """Legacy function for backward compatibility"""
    return InferenceEngine.nz_inference(key, value, key_cache, value_cache, slot_map)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.float16, np.int8, bfloat16])
@pytest.mark.parametrize('kv_dim', [2, 3])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reshape_and_cache_nd_key_value(np_dtype, kv_dim, run_mode):
    """
    Feature: Test ReshapeAndCache.
    Description: Test ND format with key and value.
    Expectation: Assert that results are consistent with numpy.
    """
    test_config = TestConfig(device_target="Ascend", mode=run_mode)
    test_config.apply()
    
    net = ReshapeAndCacheAll()

    np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nd_inputs(
        np_dtype, kv_dim)
    np_k_cache_out, np_v_cache_out = nd_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
    
    # Run test
    _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map)
    TestResultVerifier.verify_results(ms_k_cache, np_k_cache_out, np_dtype)
    TestResultVerifier.verify_results(ms_v_cache, np_v_cache_out, np_dtype)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.float16, np.int8, bfloat16])
@pytest.mark.parametrize('kv_dim', [2, 3])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reshape_and_cache_nd_key(np_dtype, kv_dim, run_mode):
    """
    Feature: Test ReshapeAndCache.
    Description: Test ND format with key only.
    Expectation: Assert that results are consistent with numpy.
    """
    test_config = TestConfig(device_target="Ascend", mode=run_mode)
    test_config.apply()
    
    net = ReshapeAndCacheKey()

    np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nd_inputs(
        np_dtype, kv_dim)
    np_k_cache_out, _ = nd_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
    
    # Run test
    _ = net(ms_k, key_cache=ms_k_cache, slot_map=ms_slot_map)
    TestResultVerifier.verify_results(ms_k_cache, np_k_cache_out, np_dtype)
