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
import mindspore.ops as ops

class Model(ms.nn.Cell):
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.margin = margin
        self.maximum = ops.Maximum()
        self.mean = ops.ReduceMean()

    def construct(self, anchor, positive, negative):
        distance_ap = ops.norm(anchor - positive, dim=1)
        distance_an = ops.norm(anchor - negative, dim=1)
        losses = self.maximum(distance_ap - distance_an + self.margin, 0.0)
        return self.mean(losses)

batch_size = 128
input_shape = (4096,)
dim = 1

def get_inputs():
    return [ops.randn(batch_size, *input_shape), 
            ops.randn(batch_size, *input_shape), 
            ops.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]