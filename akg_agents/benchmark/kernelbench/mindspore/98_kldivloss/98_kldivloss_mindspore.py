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
from mindspore import nn, ops


class Model(nn.Cell):
    

    def __init__(self):
        super(Model, self).__init__()

    def construct(self, predictions, targets):
        return ops.kl_div(ms.ops.log(predictions), targets, reduction='batchmean')


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    return [ms.ops.randn(batch_size, *input_shape).softmax(axis=-1),
            ms.ops.randn(batch_size, *input_shape).softmax(axis=-1)]


def get_init_inputs():
    return []
