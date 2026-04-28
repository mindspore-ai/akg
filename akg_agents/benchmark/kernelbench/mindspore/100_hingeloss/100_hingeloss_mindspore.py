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

import mindspore as ms
from mindspore import nn


class Model(nn.Cell):
    

    def __init__(self):
        super(Model, self).__init__()

    def construct(self, predictions, targets):
        return ms.ops.mean(ms.ops.clamp(1 - predictions.astype(ms.float32) * targets.astype(ms.float32), min=0))


batch_size = 128
input_shape = (1,)
dim = 1


def get_inputs():
    ms.set_seed(42)
    return [
        ms.ops.randn(batch_size, *input_shape).astype(ms.float32),
        (ms.ops.randint(0, 2, (batch_size, 1)).float() * 2 - 1).astype(ms.float32)
    ]


def get_init_inputs():
    return []
