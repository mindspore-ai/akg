# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
# limitations under the License

"""
fused operator dsl function: fused_relu_grad
ResNet50 fused_computation.251 in XLA patterns
"""
from __future__ import absolute_import
import akg.topi as topi

def fused_relu_grad(input1, input2, input3, c1):
    """
    fused_relu_grad.

    Args:
        input1 ~ input3: tvm.tensor.Tensor.
        c1: const.

    Returns:
        Three output (list of tvm.tensor.Tensor).
    """
    data_zero = topi.full_like(input3, c1)
    cmp_zero = topi.greater(input3, data_zero)
    data_add = topi.add(input1, input2)

    return topi.where(cmp_zero, data_add, data_zero)