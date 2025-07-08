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
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=kernel_size, 
            stride=kernel_size if stride is None else stride,
            padding=padding
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.avg_pool(x)


batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3


def get_inputs():
    x = ms.ops.randn(batch_size, channels, height, width, dtype=ms.float32)
    return [x]


def get_init_inputs():
    return [kernel_size]
