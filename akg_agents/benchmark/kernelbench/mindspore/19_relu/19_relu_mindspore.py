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
    """
    ReLU激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        计算ReLU激活函数
        Args:
            x: 输入张量
        Returns:
            ReLU激活后的张量
        """
        return ms.ops.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = ms.ops.randn(batch_size, dim, dtype=ms.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
