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
    """方阵乘法模型"""
    
    def __init__(self):
        super().__init__()
        ms.set_seed(0)  # 添加随机种子

    def construct(self, A: ms.Tensor, B: ms.Tensor) -> ms.Tensor:
        """前向计算"""
        if A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError("Inputs must be square matrices of same size")
        return ms.ops.matmul(A, B)


N = 2048


def get_inputs():
    A = ms.ops.randn(N, N, dtype=ms.float16)
    B = ms.ops.randn(N, N, dtype=ms.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
