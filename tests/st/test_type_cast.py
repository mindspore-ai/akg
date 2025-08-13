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

import numpy as np
import mindspore as ms
from mindspore import Tensor, context
import pytest
import ms_custom_ops


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_custom_type_cast_int8_to_qint4x2(exec_mode):
    """
    Feature: Test type_cast.
    Description: Test int8 cast to qint4x2.
    Expectation: Assert that results are consistent with expected.
    """
    ms.set_device("Ascend")

    def type_cast_custom(x, dtype):
        return ms_custom_ops.type_cast(x, dtype)

    if exec_mode == context.GRAPH_MODE:
        type_cast_custom = ms.jit(type_cast_custom, jit_level="O0", infer_boost="on")

    x_np = np.random.randint(-5, 5, size=(32, 32)).astype(np.int8)
    x_int4_np = x_np.reshape(-1) & 0x000F
    x_int4_np = x_int4_np[0::2] | (x_int4_np[1::2] << 4)
    x_int4_np = x_int4_np.reshape(32, 16)
    x_int8 = Tensor(x_int4_np, ms.int8)
    x_int4 = type_cast_custom(x_int8, ms.qint4x2)

    assert x_int8.dtype == ms.int8
    assert x_int4.dtype == ms.qint4x2
    np.testing.assert_allclose(x_int4.asnumpy(), x_int8.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_custom_type_cast_qint4x2_to_int8(exec_mode):
    """
    Feature: Test type_cast.
    Description: Test qint4x2 cast to int8.
    Expectation: Assert that results are consistent with expected.
    """
    ms.set_device("Ascend")

    def type_cast_custom(x, dtype):
        return ms_custom_ops.type_cast(x, dtype)

    if exec_mode == context.GRAPH_MODE:
        type_cast_custom = ms.jit(type_cast_custom)

    x_np = np.random.randint(-5, 5, size=(16, 64)).astype(np.int8)
    x_int4_np = x_np.reshape(-1) & 0x000F
    x_int4_np = x_int4_np[0::2] | (x_int4_np[1::2] << 4)
    x_int4_np = x_int4_np.reshape(16, 32)
    x_int4 = Tensor(x_int4_np, ms.qint4x2)
    x_int8 = type_cast_custom(x_int4, ms.int8)

    assert x_int8.dtype == ms.int8
    assert x_int4.dtype == ms.qint4x2
    np.testing.assert_allclose(x_int4.asnumpy(), x_int8.asnumpy())
