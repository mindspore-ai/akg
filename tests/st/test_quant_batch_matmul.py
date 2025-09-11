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

""" tests quant_batch_matmul """

import pytest
from functools import wraps
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
import ms_custom_ops


def jit(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        if ms.get_context("mode") == "PYNATIVE_MODE":
            return func(*args, **kwargs)
        return ms.jit(func, jit_level="O0", infer_boost="on")(*args, **kwargs)
    return decorator

def trans_quant_param(scale, shape):
    scale_uint32 = np.frombuffer(scale, np.uint32).reshape(shape)
    # 与高19位运算，模拟硬件
    scale_uint32 &= 0XFFFFE000
    scale_uint64 = np.zeros(shape, np.uint64)
    scale_uint64 |= np.uint64(scale_uint32)
    scale_uint64 |= (1 << 46)
    scale_int64 = np.int64(scale_uint64)
    return scale_int64

class QuantBatchMatmulNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.quant_batch_matmul = ms_custom_ops.quant_batch_matmul

    @jit
    def construct(self, x1, x2, scale, offset=None, bias=None, pertoken_scale=None,
                  transpose_x1=False, transpose_x2=False, x2_format="ND", output_dtype=ms.float16):
        out = self.quant_batch_matmul(x1, x2, scale, offset, bias, pertoken_scale,
                                      transpose_x1, transpose_x2, x2_format, output_dtype)
        return out

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.bfloat16, ms.int32])
def test_custom_quant_batch_matmul_basic(exec_mode, dtype):
    """
    Feature: Test quant_batch_matmul basic functionality.
    Description: Test quant_batch_matmul operation.
    Expectation: Assert that results are consistent with expected.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    quant_batch_matmul = QuantBatchMatmulNet()

    m = 128
    k = 256
    n = 128
    x1 = np.random.randint(-5, 5, size=(m, k)).astype(np.int8)
    x2 = np.random.randint(-5, 5, size=(k, n)).astype(np.int8)
    scale = np.ones([1]).astype(np.float32)
    expected = np.matmul(x1.astype(np.int32), x2.astype(np.int32)) * scale
    output = quant_batch_matmul(Tensor(x1), Tensor(x2), Tensor(scale), output_dtype=dtype)

    assert np.allclose(expected, output.astype(ms.float32).asnumpy(), 0.01)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
def test_custom_quant_batch_matmul_bfp16_nz():
    """
    Feature: Test quant_batch_matmul basic functionality.
    Description: Test quant_batch_matmul operation.
    Expectation: Assert that results are consistent with expected.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE)
    quant_batch_matmul = QuantBatchMatmulNet()

    batch = 2
    m = 128
    k = 256
    n = 128
    x1 = np.random.randint(-5, 5, size=(batch, m, k)).astype(np.int8)
    x2 = np.random.randint(-5, 5, size=(batch, k, n)).astype(np.int8)
    scale = np.ones([n]).astype(np.float32)
    expected = np.matmul(x1.astype(np.int32), x2.astype(np.int32)) * scale

    x1_dyn = Tensor(shape=[None, None, None], dtype=ms.int8)
    x2_dyn = Tensor(shape=[None, None, None], dtype=ms.int8)
    scale_dyn = Tensor(shape=[None], dtype=ms.float32)
    quant_batch_matmul.set_inputs(x1_dyn, x2_dyn, scale_dyn, None, None, None, False, False,
                                  "FRACTAL_NZ", ms.bfloat16)

    ms_x1 = Tensor(x1)
    ms_x2 = Tensor(x2)
    ms_x2 = ms_custom_ops.trans_data(ms_x2, transdata_type=1)
    ms_scale = Tensor(scale)
    output = quant_batch_matmul(ms_x1, ms_x2, ms_scale, x2_format="FRACTAL_NZ", output_dtype=ms.bfloat16)

    assert np.allclose(expected, output.astype(ms.float32).asnumpy(), 0.01)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.bfloat16, ms.float16])
def test_custom_quant_batch_matmul_pertoken(exec_mode, dtype):
    """
    Feature: Test quant_batch_matmul basic functionality.
    Description: Test quant_batch_matmul operation with pertoken.
    Expectation: Assert that results are consistent with expected.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    quant_batch_matmul = QuantBatchMatmulNet()

    m = 64
    k = 512
    n = 128
    x1 = np.random.randint(-5, 5, size=(m, k)).astype(np.int8)
    x2 = np.random.randint(-5, 5, size=(k, n)).astype(np.int8)
    scale = np.ones([n]).astype(np.float32)
    pertoken_scale = np.random.randn(m).astype(np.float32)
    expected = np.matmul(x1.astype(np.int32), x2.astype(np.int32)) * scale * pertoken_scale.reshape(-1, 1)

    ms_x1 = Tensor(x1)
    ms_x2 = Tensor(x2)
    ms_scale = Tensor(scale)
    ms_pertoken_scale = Tensor(pertoken_scale)
    output = quant_batch_matmul(ms_x1, ms_x2, ms_scale, pertoken_scale=ms_pertoken_scale, output_dtype=dtype)

    assert np.allclose(expected, output.astype(ms.float32).asnumpy(), 0.01)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_custom_quant_batch_matmul_with_transpose(exec_mode):
    """
    Feature: Test quant_batch_matmul with transpose parameters.
    Description: Test quant_batch_matmul operation with transpose_x1 and transpose_x2 set to True.
    Expectation: Assert that results are consistent with expected shape.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    quant_batch_matmul = QuantBatchMatmulNet()

    batch = 8
    m = 32
    k = 64
    n = 512
    x1 = np.random.randint(-5, 5, size=(batch, k, m)).astype(np.int8)
    x2 = np.random.randint(-5, 5, size=(batch, n, k)).astype(np.int8)
    scale = np.random.randn(1).astype(np.float32)
    np_x1 = x1.astype(np.int32).transpose(0, 2, 1)
    np_x2 = x2.astype(np.int32).transpose(0, 2, 1)
    expected = np.matmul(np_x1, np_x2) * scale

    ms_x1 = Tensor(x1)
    ms_x2 = Tensor(x2)
    ms_scale = Tensor(scale)
    output = quant_batch_matmul(ms_x1, ms_x2, ms_scale, transpose_x1=True, transpose_x2=True, output_dtype=ms.bfloat16)

    assert np.allclose(expected, output.astype(ms.float32).asnumpy(), 0.01)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_custom_quant_batch_matmul_with_scale_int64(exec_mode):
    """
    Feature: Test quant_batch_matmul with transpose parameters.
    Description: Test quant_batch_matmul operation with scale int64.
    Expectation: Assert that results are consistent with expected shape.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    quant_batch_matmul = QuantBatchMatmulNet()

    batch = 8
    m = 16
    k = 256
    n = 96
    x1 = np.random.randint(-5, 5, size=(batch, m, k)).astype(np.int8)
    x2 = np.random.randint(-5, 5, size=(batch, k, n)).astype(np.int8)
    scale = np.random.randn(n).astype(np.float32)
    scale_int64 = trans_quant_param(scale, (n,))
    expected = np.matmul(x1.astype(np.int32), x2.astype(np.int32)) * scale
    output = quant_batch_matmul(Tensor(x1), Tensor(x2), Tensor(scale_int64), output_dtype=ms.float16)

    assert np.allclose(expected, output.astype(ms.float32).asnumpy(), 0.01)
