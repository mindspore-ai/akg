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

"""mean_square"""
from test_op.square import square
from akg.ops.math.mean import mean


def mean_square(inputs, axis=None, keepdims=False):
    """Mean of square value of a tensor, alongside the specified axis.

    Arguments:
        input: A tensor.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.

    Returns:
        A tensor with the mean of element-wise square value of `input`.

    Notice: There is some precision problem for the operator and remain to solve
    """
    inputs_square = square(inputs)
    return mean(inputs_square, axis, keepdims)
