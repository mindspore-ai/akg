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
import mindspore.nn as nn
import numpy as np


class Model(nn.Cell):
    

    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(
            normalized_shape=normalized_shape,
            begin_norm_axis=1, 
            begin_params_axis=1
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.ln(x)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = ms.Tensor(np.random.randn(batch_size, features, dim1, dim2), ms.float32)
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]