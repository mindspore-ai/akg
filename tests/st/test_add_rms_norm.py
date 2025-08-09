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

@ms.jit(jit_level="O0", infer_boost="on", backend="ms_backend")
def add_rms_norm(x1, x2, gamma, epsilon=1e-6):
    return ms.ops.add_rms_norm(x1, x2, gamma, epsilon)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32, ms.bfloat16])
@pytest.mark.parametrize('shape', [(1, 1024, 1024)])
def test_custom_add_rms_norm(exec_mode, dtype, shape):
    """
    Feature: Test add_rms_norm.
    Description: Test add_rms_norm.
    Expectation: Assert that results are consistent with expected.
    """
    ms.set_device("Ascend")

    def add_rms_norm_custom(x1, x2, gamma, epsilon=1e-6):
        return ms_custom_ops.add_rms_norm(x1, x2, gamma, epsilon)

    if exec_mode == context.GRAPH_MODE:
        add_rms_norm_custom = ms.jit(add_rms_norm_custom, jit_level="O0", infer_boost="on", backend="ms_backend")

    x1 = Tensor(np.random.rand(*shape), dtype)
    x2 = Tensor(np.random.rand(*shape), dtype)
    gamma = Tensor(np.random.rand(*shape), dtype)
    eps = 1e-6
    out = add_rms_norm_custom(x1, x2, gamma, eps)
    expect = add_rms_norm(x1, x2, gamma, eps)
    np.testing.assert_allclose(
        out[0].astype(ms.float32).asnumpy(),
        expect[0].astype(ms.float32).asnumpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        out[2].astype(ms.float32).asnumpy(),
        expect[2].astype(ms.float32).asnumpy(),
        rtol=1e-3,
        atol=1e-3,
    )
