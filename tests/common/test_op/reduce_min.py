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

"""operator dsl function: reduce_min"""
from akg.ops.math import reduce_min_max_common


def reduce_min(data, axis=None, keepdims=False):
    """
    Computes the minimum of elements over a given axis or a list of axes of a tensor.

    Args:
        data: The input tensor to reduce. Should be a akg.tvm.Tensor of type float16, float32, int8, uint8, int32.
        axis: The dimensions to reduce. Could be None(the default), int, list or tuple.
              If None, all dimensions will be reduced.
              If int or list, must be in the range [-len(data.shape), len(data.shape)- 1]
        keepdims: Boolean. If True, retains reduced dimensions with length 1, default value is False

    Returns:
        akg.tvm.Tensor of same type as input tensor data.
    """

    return reduce_min_max_common.reduce_min_max(data, axis=axis, keepdims=keepdims, method="min")
