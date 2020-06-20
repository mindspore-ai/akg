# Copyright 2019 Huawei Technologies Co., Ltd
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

"""sum_square"""
from test_op.square import square
from akg.ops.math.sum import sum_value

def sum_square(inputs, axis=None, keepdims=False):
    """
    Computes the sum of square value of input tensor along axis.

    Args:
        input: The input akg.tvm.tensor.
        axis: An integer, specifies the dimensions to reduce when performing the sum operation.
        keepdims: A boolean, if True, retains reduced dimensions with length 1, default value is False.

    Returns:
        A akg.tvm.Tensor of same type as input.
    """

    inputs_square = square(inputs)
    return sum_value(inputs_square, axis, keepdims)
