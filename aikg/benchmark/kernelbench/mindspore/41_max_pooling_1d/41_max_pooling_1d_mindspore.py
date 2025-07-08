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
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False):
        super(Model, self).__init__()
        self.maxpool1d = nn.MaxPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            pad_mode='pad')

    def construct(self, x):
        return self.maxpool1d(x)


def get_inputs():
    return ms.Tensor(np.random.randn(16, 64, 128), ms.float32)

def get_init_inputs():
    return [4, 2, 2, 3, False]