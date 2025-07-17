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
from mindspore.ops import ModuleWrapper
from mindspore import Tensor, context, Parameter, ops
import pytest
import ms_custom_ops

@pytest.mark.parametrize('exec_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('np_dtype', [np.float16])
def test_custom_add(exec_mode, np_dtype):
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    
    class MyNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y):
            return ms_custom_ops.add(x, y)
    
    x = np.random.randn(4, 2048).astype(np_dtype)
    y = np.random.randn(4, 2048).astype(np_dtype)
    net = MyNet()
    out = net(Tensor(x), Tensor(y))
    expect = x + y
    np.testing.assert_allclose(out.asnumpy(), expect, rtol=1e-3, atol=1e-3)
