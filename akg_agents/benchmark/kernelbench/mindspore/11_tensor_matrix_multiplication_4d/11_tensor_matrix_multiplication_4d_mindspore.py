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

    def construct(self, A, B):
        
        return ms.ops.matmul(A, B)


# Test code
b = 16
i = 256
j = 512
l = 256
k = 768


def get_inputs():
    A = ms.ops.randn(b, i, j, l, dtype=ms.float16)
    B = ms.ops.randn(l, k, dtype=ms.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
