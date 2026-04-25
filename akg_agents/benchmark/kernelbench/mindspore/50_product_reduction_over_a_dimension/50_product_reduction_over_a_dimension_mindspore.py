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
    

    def __init__(self, dim: int):
        
        super(Model, self).__init__()
        self.dim = dim

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        
        return ms.ops.prod(x, axis=self.dim)


batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1


def get_inputs():
    x = ms.ops.randn(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [reduction_dim]
