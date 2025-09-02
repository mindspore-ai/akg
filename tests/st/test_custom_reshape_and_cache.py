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
from functools import cache, wraps
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

def jit_for_graph_mode(fn):
    """
    A decorator that conditionally applies jit to a function at runtime based on the context mode.
    """
    jitted_fn = jit(fn)
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if context.get_context("mode") == context.GRAPH_MODE:
            return jitted_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper

# Global constants
NUM_SLOTS = 20
SLOT_SIZE = 64
BATCH_SIZE = 13
SEQ_LEN = 3
NUM_HEADS = 16
K_HEAD_DIM = 32
V_HEAD_DIM = 32


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
    
    @jit_for_graph_mode
    def construct(self, key, value, key_cache, value_cache, slot_map, cache_mode, head_num=0):
        return ms_custom_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slot_map, cache_mode, head_num)


class ReshapeAndCacheKey(nn.Cell):
    """Reshape and cache operation for NZ/ND format with key only"""
    
    @jit_for_graph_mode
    def construct(self, key, key_cache, slot_map, cache_mode):
        return ms_custom_ops.reshape_and_cache(
            key, key_cache=key_cache, slot_mapping=slot_map, cache_mode=cache_mode)


class MindSporeInputFactory:
    """Factory for creating MindSpore inputs"""
    
    @staticmethod
    def create_inputs(np_k: np.ndarray, np_v: np.ndarray, 
                     np_k_cache: np.ndarray, np_v_cache: np.ndarray, 
                     np_slot_map: np.ndarray) -> Tuple[Tensor, ...]:
        """Create MindSpore inputs"""
        ms_key = Tensor(np_k)
        ms_value = Tensor(np_v)
        ms_key_cache = Tensor(np_k_cache)
        ms_value_cache = Tensor(np_v_cache)
        ms_slot_map = Tensor(np_slot_map)
        return ms_key, ms_value, ms_key_cache, ms_value_cache, ms_slot_map


def create_ms_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map):
    """Legacy function for backward compatibility"""
    return MindSporeInputFactory.create_inputs(np_k, np_v, np_k_cache, np_v_cache, np_slot_map)


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
        ms.set_device(self.device_target)
        context.set_context(mode=self.mode)
        if self.jit_config:
            context.set_context(jit_config=self.jit_config)


class DimensionTestHelper:
    """Helper class for testing different dimension combinations"""
    
    @staticmethod
    def run_with_dimensions(k_head_dim: int, v_head_dim: int, test_func):
        """Run test with specified dimensions and restore original values"""
        global K_HEAD_DIM, V_HEAD_DIM
        original_k_head_dim = K_HEAD_DIM
        original_v_head_dim = V_HEAD_DIM
        
        try:
            K_HEAD_DIM = k_head_dim
            V_HEAD_DIM = v_head_dim
            test_func()
        finally:
            K_HEAD_DIM = original_k_head_dim
            V_HEAD_DIM = original_v_head_dim


# ===============================
#        RESHAPE AND CACHE TEST ARCHITECTURE
# ===============================
"""
Test Structure Overview:

1. ND FORMAT TESTS (cache_mode=0):
   - Direct ND format testing without format conversion
   - Data flow: Input(ND) → ReshapeAndCache → Output(ND) → Verify
   - Tests: test_reshape_and_cache_nd_*

2. NZ FORMAT TESTS (cache_mode=1): 
   - Tests FRACTAL_NZ format with format conversion using trans_data
   - Data flow: Input(ND) → TransData(ND→NZ) → ReshapeAndCache → TransData(NZ→ND) → Verify
   - Tests: test_reshape_and_cache_nz_*
   
3. KEY COMPONENTS:
   - create_nd_inputs(): Generate ND format test data
   - create_nz_inputs(): Generate NZ-compatible test data (different layout)
   - get_nd_cached_slots(): Extract verification data from ND format cache
   - get_nz_cached_slots(): Extract verification data from NZ format cache (legacy)
   - nd_inference()/nz_inference(): Generate golden reference results

4. VERIFICATION STRATEGY:
   - ND tests: Both actual and golden use ND format → direct comparison
   - NZ tests: Convert actual results back to ND format → compare with ND golden
"""

# ===============================
#        ND FORMAT TESTS
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
    def get_update_shapes(kv_dim: int, k_head_dim=None, v_head_dim=None) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        """Get update shapes for key and value, and number of tokens based on dimension"""
        # Use provided dimensions or fall back to global constants
        actual_k_head_dim = k_head_dim if k_head_dim is not None else K_HEAD_DIM
        actual_v_head_dim = v_head_dim if v_head_dim is not None else V_HEAD_DIM
        
        if kv_dim == 2:
            key_update_shape = (BATCH_SIZE * SEQ_LEN, NUM_HEADS * actual_k_head_dim)
            value_update_shape = (BATCH_SIZE * SEQ_LEN, NUM_HEADS * actual_v_head_dim)
            num_tokens = key_update_shape[0]
        elif kv_dim == 3:
            key_update_shape = (BATCH_SIZE, SEQ_LEN, NUM_HEADS * actual_k_head_dim)
            value_update_shape = (BATCH_SIZE, SEQ_LEN, NUM_HEADS * actual_v_head_dim)
            num_tokens = key_update_shape[0] * key_update_shape[1]
        else:
            raise ValueError(f"Key's dim should be 2 or 3, but got {kv_dim}")
        return key_update_shape, value_update_shape, num_tokens
    
    @staticmethod
    def get_update_shape(kv_dim: int, is_key: bool = True, k_head_dim=None, v_head_dim=None) -> Tuple[Tuple[int, ...], int]:
        """Legacy method for backward compatibility"""
        key_shape, value_shape, num_tokens = TestDataGenerator.get_update_shapes(kv_dim, k_head_dim, v_head_dim)
        return (key_shape if is_key else value_shape), num_tokens


class NDDataGenerator(TestDataGenerator):
    """Data generator for ND format"""
    
    @staticmethod
    def create_inputs(dtype: np.dtype, kv_dim: int, k_head_dim=None, v_head_dim=None) -> Tuple[np.ndarray, ...]:
        """Create ND format inputs"""
        # Use provided dimensions or fall back to global constants
        actual_k_head_dim = k_head_dim if k_head_dim is not None else K_HEAD_DIM
        actual_v_head_dim = v_head_dim if v_head_dim is not None else V_HEAD_DIM
        
        key_cache_shape = (NUM_SLOTS, SLOT_SIZE, NUM_HEADS, actual_k_head_dim)
        value_cache_shape = (NUM_SLOTS, SLOT_SIZE, NUM_HEADS, actual_v_head_dim)
        key_update_shape, value_update_shape, num_tokens = TestDataGenerator.get_update_shapes(kv_dim, k_head_dim, v_head_dim)
        
        key_update = TestDataGenerator.create_random_data(key_update_shape, dtype)
        value_update = TestDataGenerator.create_random_data(value_update_shape, dtype)
        key_cache = TestDataGenerator.create_random_data(key_cache_shape, dtype)
        value_cache = TestDataGenerator.create_random_data(value_cache_shape, dtype)
        slot_map = TestDataGenerator.create_slot_map(num_tokens)
        
        return key_update, value_update, key_cache, value_cache, slot_map


def create_nd_inputs(dtype=np.float16, kv_dim=3, k_head_dim=None, v_head_dim=None):
    """Legacy function for backward compatibility"""
    return NDDataGenerator.create_inputs(dtype, kv_dim, k_head_dim, v_head_dim)


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
        
        # Use different dimensions for key and value
        key_head = key_cache.shape[2]
        key_head_dim = key_cache.shape[3]
        value_head = value_cache.shape[2]
        value_head_dim = value_cache.shape[3]
        
        key_tmp = key_tmp.reshape(-1, key_head, key_head_dim)
        value_tmp = value_tmp.reshape(-1, value_head, value_head_dim)
        
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
        
        # Use different dimensions for key and value
        key_tmp = key_tmp.reshape(-1, key_cache.shape[2])
        value_tmp = value_tmp.reshape(-1, value_cache.shape[2])
        
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
    _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, 0)
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
    test_config = TestConfig(device_target="Ascend", mode=run_mode,
                           jit_config={"jit_level": "O0"})
    test_config.apply()
    
    net = ReshapeAndCacheKey()

    np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nd_inputs(
        np_dtype, kv_dim)
    np_k_cache_out, _ = nd_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
    
    # Run test
    _ = net(ms_k, key_cache=ms_k_cache, slot_map=ms_slot_map, cache_mode=0)
    TestResultVerifier.verify_results(ms_k_cache, np_k_cache_out, np_dtype)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('np_dtype', [np.float16, np.int8, bfloat16])
@pytest.mark.parametrize('kv_dim', [2, 3])
@pytest.mark.parametrize('k_head_dim', [32, 64, 128])
@pytest.mark.parametrize('v_head_dim', [32, 64, 128])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reshape_and_cache_nd_key_value_different_dimensions(np_dtype, kv_dim, k_head_dim, v_head_dim, run_mode):
    """
    Feature: Test ReshapeAndCache.
    Description: Test ND format with different K_HEAD_DIM and V_HEAD_DIM combinations.
    Expectation: Assert that results are consistent with numpy.
    """
    def run_test():
        test_config = TestConfig(device_target="Ascend", mode=run_mode)
        test_config.apply()
        
        net = ReshapeAndCacheAll()

        np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nd_inputs(
            np_dtype, kv_dim, k_head_dim, v_head_dim)
        np_k_cache_out, np_v_cache_out = nd_inference(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

        ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
        
        # Run test
        _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, 0)
        TestResultVerifier.verify_results(ms_k_cache, np_k_cache_out, np_dtype)
        TestResultVerifier.verify_results(ms_v_cache, np_v_cache_out, np_dtype)
    
    DimensionTestHelper.run_with_dimensions(k_head_dim, v_head_dim, run_test)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('kv_dim', [2, 3])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reshape_and_cache_nz_different_key_value_dimensions(kv_dim, run_mode):
    """
    Feature: Test ReshapeAndCache with FRACTAL_NZ format and different key/value dimensions.
    Description: Test with very different K_HEAD_DIM(96) and V_HEAD_DIM(16) using trans_data conversion.
    Test Flow: ND → trans_data(ND→NZ) → ReshapeAndCache(cache_mode=1) → trans_data(NZ→ND) → Verify
    Expectation: Handles dimension differences correctly after roundtrip FRACTAL_NZ conversion.
    """
    def run_test():
        # Setup context
        jit_config = {"jit_level": "O0"}
        test_config = TestConfig(device_target="Ascend", mode=run_mode, jit_config=jit_config)
        test_config.apply()
        
        net = ReshapeAndCacheAll()

        np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nz_inputs(
            np.float16, np.float16, kv_dim)
        
        # Verify that key and value have different shapes
        assert np_k.shape != np_v.shape, f"Key and value should have different shapes: {np_k.shape} vs {np_v.shape}"
        assert np_k_cache.shape != np_v_cache.shape, f"Key and value cache should have different shapes: {np_k_cache.shape} vs {np_v_cache.shape}"
        
        np_k_cache_out, np_v_cache_out = nz_inference(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

        # Create MindSpore inputs with appropriate format
        ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
        # Convert ND to FRACTAL_NZ format using trans_data
        ms_k_cache = ms_custom_ops.trans_data(ms_k_cache, transdata_type=1)  # ND_TO_FRACTAL_NZ
        ms_v_cache = ms_custom_ops.trans_data(ms_v_cache, transdata_type=1)  # ND_TO_FRACTAL_NZ

        _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, cache_mode=1, head_num=NUM_HEADS)
       
        # Extract and verify results - both use ND format extraction
        ms_k_cache_np = ms_k_cache.asnumpy()
        ms_v_cache_np = ms_v_cache.asnumpy()
        
        ms_k_output = get_nz_cached_slots(ms_k_cache_np, np_slot_map)
        golden_k_output = get_nd_cached_slots(np_k_cache_out, np_slot_map)  # Golden is already ND format
        
        ms_v_output = get_nz_cached_slots(ms_v_cache_np, np_slot_map)
        golden_v_output = get_nd_cached_slots(np_v_cache_out, np_slot_map)  # Golden is already ND format

        # Verify results
        assert np.allclose(ms_k_output, golden_k_output, 0.001, 0.001)
        assert np.allclose(ms_v_output, golden_v_output, 0.001, 0.001)
    
    # Test with very different dimensions
    DimensionTestHelper.run_with_dimensions(96, 16, run_test)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('kv_dim', [2, 3])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reshape_and_cache_different_key_value_dimensions(kv_dim, run_mode):
    """
    Feature: Test ReshapeAndCache.
    Description: Test with significantly different K_HEAD_DIM and V_HEAD_DIM.
    Expectation: Assert that results are consistent with numpy.
    """
    def run_test():
        test_config = TestConfig(device_target="Ascend", mode=run_mode)
        test_config.apply()
        
        net = ReshapeAndCacheAll()

        # Test with very different dimensions
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nd_inputs(
            np.float16, kv_dim)
        
        # Verify that key and value have different shapes
        assert np_k.shape != np_v.shape, f"Key and value should have different shapes: {np_k.shape} vs {np_v.shape}"
        assert np_k_cache.shape != np_v_cache.shape, f"Key and value cache should have different shapes: {np_k_cache.shape} vs {np_v_cache.shape}"
        
        np_k_cache_out, np_v_cache_out = nd_inference(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

        ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
        
        # Run test
        _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, cache_mode=0)
        TestResultVerifier.verify_results(ms_k_cache, np_k_cache_out, np.float16)
        TestResultVerifier.verify_results(ms_v_cache, np_v_cache_out, np.float16)
    
    # Test with very different dimensions
    DimensionTestHelper.run_with_dimensions(128, 32, run_test)


# ===============================
#        NZ FORMAT TESTS (FRACTAL_NZ)
# ===============================
"""
NZ Format Test Flow:
1. Create initial ND format cache tensors
2. Convert cache tensors to FRACTAL_NZ format using trans_data(type=2)
3. Run ReshapeAndCache with cache_mode=1 (NZ format mode)
4. Convert results back to ND format using trans_data(type=1) for verification
5. Compare with golden ND results using get_nd_cached_slots()

Note: The 'NZ' in test names refers to FRACTAL_NZ format compatibility,
but all verification is done in ND format after conversion back.
"""
def convert_cache_nz_to_nd_and_verify(ms_k_cache, ms_v_cache, np_k_cache_out, np_v_cache_out, 
                                     np_slot_map, k_dtype, v_dtype):
    """
    Helper function to convert FRACTAL_NZ cache results back to ND format and perform verification.
    This eliminates code duplication across NZ test functions.
    """
    # Convert FRACTAL_NZ cache results back to ND format for verification
    ms_k_cache_nd = ms_custom_ops.trans_data(ms_k_cache, transdata_type=0)  # FRACTAL_NZ_TO_ND
    ms_v_cache_nd = ms_custom_ops.trans_data(ms_v_cache, transdata_type=0)  # FRACTAL_NZ_TO_ND
    
    # Extract and verify results - convert to numpy arrays
    ms_k_cache_np = ms_k_cache_nd.asnumpy()
    ms_v_cache_np = ms_v_cache_nd.asnumpy()
    
    # Handle bfloat16 conversion
    if k_dtype == bfloat16:
        ms_k_cache_np = ms_k_cache_np.astype(np.float32)
        np_k_cache_out = np_k_cache_out.astype(np.float32)
    
    if v_dtype == bfloat16:
        ms_v_cache_np = ms_v_cache_np.astype(np.float32)
        np_v_cache_out = np_v_cache_out.astype(np.float32)
    
    # Extract cached slots for verification - both use ND format extraction
    ms_k_output = get_nd_cached_slots(ms_k_cache_np, np_slot_map)
    golden_k_output = get_nd_cached_slots(np_k_cache_out, np_slot_map)  # Golden is already ND format
    
    ms_v_output = get_nd_cached_slots(ms_v_cache_np, np_slot_map)
    golden_v_output = get_nd_cached_slots(np_v_cache_out, np_slot_map)  # Golden is already ND format
    
    # Verify results
    assert np.allclose(ms_k_output, golden_k_output, 0.001, 0.001), \
        f"Key cache mismatch: max_diff={np.max(np.abs(ms_k_output - golden_k_output))}"
    assert np.allclose(ms_v_output, golden_v_output, 0.001, 0.001), \
        f"Value cache mismatch: max_diff={np.max(np.abs(ms_v_output - golden_v_output))}"


class NZDataGenerator(TestDataGenerator):
    """Data generator for NZ format"""
    
    @staticmethod
    def create_inputs(k_dtype: np.dtype, v_dtype: np.dtype, kv_dim: int, k_head_dim=None, v_head_dim=None) -> Tuple[np.ndarray, ...]:
        """Create NZ format inputs"""
        # Use provided dimensions or fall back to global constants
        actual_k_head_dim = k_head_dim if k_head_dim is not None else K_HEAD_DIM
        actual_v_head_dim = v_head_dim if v_head_dim is not None else V_HEAD_DIM
        
        k_cache_shape = (NUM_SLOTS, SLOT_SIZE, NUM_HEADS * actual_k_head_dim)
        v_cache_shape = (NUM_SLOTS, SLOT_SIZE, NUM_HEADS * actual_v_head_dim)
        key_update_shape, value_update_shape, num_tokens = TestDataGenerator.get_update_shapes(kv_dim, k_head_dim, v_head_dim)
        
        key_update = TestDataGenerator.create_random_data(key_update_shape, k_dtype)
        value_update = TestDataGenerator.create_random_data(value_update_shape, v_dtype)
        key_cache = np.zeros(k_cache_shape, dtype=k_dtype)
        value_cache = np.zeros(v_cache_shape, dtype=v_dtype)
        slot_map = TestDataGenerator.create_slot_map(num_tokens)
        
        return key_update, value_update, key_cache, value_cache, slot_map


def create_nz_inputs(k_dtype=np.float16, v_dtype=np.float16, kv_dim=3, k_head_dim=None, v_head_dim=None):
    """Legacy function for backward compatibility"""
    return NZDataGenerator.create_inputs(k_dtype, v_dtype, kv_dim, k_head_dim, v_head_dim)


def get_nz_cached_slots(cache, slot_map):
    ans = []

    num_slots = cache.shape[0]
    slot_size = cache.shape[1]
    hidden_size = cache.shape[2]

    if cache.dtype == np.int8:
        cache_shape = (num_slots, hidden_size // 32, slot_size, 32)
    else:
        cache_shape = (num_slots, hidden_size // 16, slot_size, 16)
    cache = cache.reshape(cache_shape)
    for i, slot in enumerate(slot_map):
        if slot < 0:
            continue
        slot_idx = slot // slot_size
        slot_offset = slot % slot_size
        tmp = []  # Reset tmp for each slot
        for j in range(cache.shape[1]):
            tmp.append(cache[slot_idx][j][slot_offset])
        ans.append(np.concatenate(tmp, axis=0))
    ans = np.concatenate(ans)
    return ans


def get_nd_cached_slots(cache, slot_map):
    ans = []
    for slot in slot_map:
        if slot < 0:
            continue
        slot_idx = slot // SLOT_SIZE
        slot_offset = slot % SLOT_SIZE
        ans.append(cache[slot_idx][slot_offset])
    ans = np.concatenate(ans)
    return ans


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('kv_dim', [2, 3])
@pytest.mark.parametrize('k_dtype', [np.float16, bfloat16, np.int8])
@pytest.mark.parametrize('v_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reshape_and_cache_nz(k_dtype, v_dtype, kv_dim, run_mode):
    """
    Feature: Test ReshapeAndCache with FRACTAL_NZ format conversion.
    Description: Test FRACTAL_NZ format compatibility using trans_data for format conversion.
    Test Flow: ND → trans_data(ND→NZ) → ReshapeAndCache(cache_mode=1) → trans_data(NZ→ND) → Verify
    Expectation: Results match golden ND format reference after roundtrip conversion.
    """
    # Skip invalid combinations
    if (k_dtype == np.float16 and v_dtype != np.float16) or \
       (k_dtype == bfloat16 and v_dtype != bfloat16):
        pytest.skip(f"Invalid combo: {k_dtype} -> {v_dtype}")
    
    # Setup context
    jit_config = {"jit_level": "O0"}
    test_config = TestConfig(device_target="Ascend", mode=run_mode, jit_config=jit_config)
    test_config.apply()
    
    net = ReshapeAndCacheAll()

    np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nz_inputs(
        k_dtype, v_dtype, kv_dim)
    np_k_cache_out, np_v_cache_out = nz_inference(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

    # Create MindSpore inputs with appropriate format
    ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
        np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
    # Convert ND to FRACTAL_NZ format using trans_data
    ms_k_cache = ms_custom_ops.trans_data(ms_k_cache, transdata_type=1)  # ND_TO_FRACTAL_NZ
    ms_v_cache = ms_custom_ops.trans_data(ms_v_cache, transdata_type=1)  # ND_TO_FRACTAL_NZ

    _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, cache_mode=1, head_num=NUM_HEADS)
 
    # Extract and verify results - convert to numpy arrays
    ms_k_cache_np = ms_k_cache.asnumpy()
    ms_v_cache_np = ms_v_cache.asnumpy()
    
    # Handle bfloat16 conversion
    if k_dtype == bfloat16:
        ms_k_cache_np = ms_k_cache_np.astype(np.float32)
        np_k_cache_out = np_k_cache_out.astype(np.float32)
    
    if v_dtype == bfloat16:
        ms_v_cache_np = ms_v_cache_np.astype(np.float32)
        np_v_cache_out = np_v_cache_out.astype(np.float32)
    
    # Extract cached slots for verification - both use ND format extraction
    ms_k_output = get_nz_cached_slots(ms_k_cache_np, np_slot_map)
    golden_k_output = get_nd_cached_slots(np_k_cache_out, np_slot_map)  # Golden is already ND format

    ms_v_output = get_nz_cached_slots(ms_v_cache_np, np_slot_map)
    golden_v_output = get_nd_cached_slots(np_v_cache_out, np_slot_map)  # Golden is already ND format
    
    # Verify results
    assert np.allclose(ms_k_output, golden_k_output, 0.001, 0.001)
    assert np.allclose(ms_v_output, golden_v_output, 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('kv_dim', [2, 3])
@pytest.mark.parametrize('k_dtype', [np.float16, bfloat16, np.int8])
@pytest.mark.parametrize('v_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('k_head_dim', [32, 64, 128])
@pytest.mark.parametrize('v_head_dim', [32, 64, 128])
def test_reshape_and_cache_nz_different_dimensions(k_dtype, v_dtype, kv_dim, run_mode, k_head_dim, v_head_dim):
    """
    Feature: Test ReshapeAndCache with FRACTAL_NZ format and various dimension combinations.
    Description: Test all combinations of K_HEAD_DIM and V_HEAD_DIM (32,64,128) using trans_data conversion.
    Test Flow: ND → trans_data(ND→NZ) → ReshapeAndCache(cache_mode=1) → trans_data(NZ→ND) → Verify
    Expectation: All dimension combinations work correctly with FRACTAL_NZ roundtrip conversion.
    """
    # Skip invalid combinations
    if (k_dtype == np.float16 and v_dtype != np.float16) or \
       (k_dtype == bfloat16 and v_dtype != bfloat16):
        pytest.skip(f"Invalid combo: {k_dtype} -> {v_dtype}")
    
    def run_test():
        # Setup context
        jit_config = {"jit_level": "O0"}
        test_config = TestConfig(device_target="Ascend", mode=run_mode, jit_config=jit_config)
        test_config.apply()
        
        net = ReshapeAndCacheAll()

        np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nz_inputs(
            k_dtype, v_dtype, kv_dim, k_head_dim, v_head_dim)
        np_k_cache_out, np_v_cache_out = nz_inference(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)

        # Create MindSpore inputs with appropriate format
        ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map = create_ms_inputs(
            np_k, np_v, np_k_cache, np_v_cache, np_slot_map)
        # Convert ND to FRACTAL_NZ format using trans_data
        ms_k_cache = ms.jit(ms_custom_ops.trans_data)(ms_k_cache, transdata_type=1)  # ND_TO_FRACTAL_NZ
        ms_v_cache = ms.jit(ms_custom_ops.trans_data)(ms_v_cache, transdata_type=1)  # ND_TO_FRACTAL_NZ

        _ = net(ms_k, ms_v, ms_k_cache, ms_v_cache, ms_slot_map, cache_mode=1, head_num=NUM_HEADS)

        # Extract and verify results - convert to numpy arrays
        # host没有FRACTAL_NZ的信息，asnumpy后还是FRACTAL_NZ格式
        ms_k_cache_np = ms_k_cache.asnumpy()
        ms_v_cache_np = ms_v_cache.asnumpy()
        
        # Handle bfloat16 conversion
        if k_dtype == bfloat16:
            ms_k_cache_np = ms_k_cache_np.astype(np.float32)
            np_k_cache_out = np_k_cache_out.astype(np.float32)
        
        if v_dtype == bfloat16:
            ms_v_cache_np = ms_v_cache_np.astype(np.float32)
            np_v_cache_out = np_v_cache_out.astype(np.float32)
        
        # Extract cached slots for verification - both use ND format extraction
        # 所以这里直接用nz格式提取
        ms_k_output = get_nz_cached_slots(ms_k_cache_np, np_slot_map)
        golden_k_output = get_nd_cached_slots(np_k_cache_out, np_slot_map)  # Golden is already ND format

        ms_v_output = get_nz_cached_slots(ms_v_cache_np, np_slot_map)
        golden_v_output = get_nd_cached_slots(np_v_cache_out, np_slot_map)  # Golden is already ND format
        
        # Verify results
        assert np.allclose(ms_k_output, golden_k_output, 0.001, 0.001)
        assert np.allclose(ms_v_output, golden_v_output, 0.001, 0.001)
    
    DimensionTestHelper.run_with_dimensions(k_head_dim, v_head_dim, run_test)
