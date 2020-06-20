#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function: reduce_max"""

import akg.tvm
from akg.ops.math import reduce_min_max_common
from akg.utils.validation_check import check_input_type


@check_input_type(akg.tvm.tensor.Tensor, (int, list, tuple, type(None)), (bool, type(None)))
def reduce_max(data, axis=None, keepdims=False):
    """
    Computes the maximum of elements over a given axis or a list of axes of a tensor.

    Args:
        data (tvm.tensor.Tensor): The input tensor to reduce. Should be of type float16, float32, int8, uint8, int32.
        axis (Union[list, tuple, int, None]): The dimensions to reduce.
                                      If None, all dimensions will be reduced.
                                      If int or list, must be in the range [-len(data.shape), len(data.shape) - 1].
        keepdims (bool): If True, retains reduced dimensions with length 1, default value is False.

    Returns:
        tvm.tensor.Tensor of same type as input tensor data.
    """
    return reduce_min_max_common.reduce_min_max(data, axis=axis, keepdims=keepdims, method="max")
