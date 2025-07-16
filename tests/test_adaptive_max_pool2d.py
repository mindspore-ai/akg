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
def test_custom_adaptive_max_pool2d(exec_mode, np_dtype):
    ms.set_device("Ascend")
    ms.set_context(mode=exec_mode)
    
    class MyNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            if exec_mode is context.PYNATIVE_MODE:
                self.adaptive_max_pool2d = ms_custom_ops.adaptive_max_pool2d
            else:
                def adaptive_max_pool2d(x, output_size):
                    mod = ModuleWrapper("custom_adaptive_max_pool2d", ms_custom_ops)
                    return mod.adaptive_max_pool2d(x, output_size)
                self.adaptive_max_pool2d = adaptive_max_pool2d

        def construct(self, x, output_size):
            return self.adaptive_max_pool2d(x, output_size)
    
    x = np.random.randn(1, 3, 32, 32).astype(np_dtype)
    output_size = (16, 16)
    net = MyNet()
    out = net(Tensor(x), output_size)
    assert out[0].shape == (1, 3, 16, 16)
