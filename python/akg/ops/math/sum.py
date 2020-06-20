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

"""operator dsl function: sum"""

import akg.topi
import akg.tvm
from akg.utils import format_transform as ft_util
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import get_shape
from akg.ops.math.cast import cast


def get_attrs():
    """get attrs."""
    attr_map = {"enable_bisect_optimize": True}
    return attr_map


@vc_util.check_input_type(akg.tvm.tensor.Tensor, (list, tuple, int, type(None)), (bool, type(None)))
def sum_value(inputs, axis=None, keepdims=False):
    """
    Computes the sum value of a tensor along the given axes.

    Args:
        inputs (tvm.tensor.Tensor): Tensor of type float16, float32.
        axis (Union[list, tuple, int, None]): Specifies which axis or axes to reduce.
        keepdims (bool): If true, the dimension specified by axis will be one.

    Returns:
        tvm.tensor.Tensor with same type as input tensor.
    """

    # Check types
    dtype = inputs.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    axis = ft_util.refine_reduce_axis(inputs, axis)
    vc_util.check_shape(inputs.shape)

    if not axis:
        output = akg.topi.identity(inputs)
    else:
        output = akg.topi.sum(inputs, axis=axis, keepdims=keepdims)
    attr_map = get_attrs()
    return output, attr_map


@vc_util.check_input_type(akg.tvm.tensor.Tensor, (list, tuple, int, type(None)), (bool, type(None)))
def sum_v2(inputs, axis=None, keepdims=True):
    """another implementation of sum with topi api."""
    dtype = inputs.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    axis = ft_util.refine_reduce_axis(inputs, axis)
    vc_util.check_shape(inputs.shape)
    if not axis:
        output = akg.topi.identity(inputs)
    else:
        if dtype == "float16":
            step_sum = cast(inputs, "float32")
        else:
            step_sum = inputs

        step_sum = akg.topi.sum(step_sum, axis=axis, keepdims=keepdims)

        if dtype == "float16":
            output = cast(step_sum, "float16")
        else:
            output = step_sum
    attr_map = get_attrs()
    return output, attr_map


def sum_by_shape(broadcast_data, original_shape):
    """sum the broadcast_data by original shape; gradient for Broadcast."""
    broadcast_shape = get_shape(broadcast_data)
    original_shape = get_shape(original_shape)
    if broadcast_shape == original_shape:
        return broadcast_data
    if original_shape == [1]:
        data, _ = sum_value(broadcast_data)
        return data

    vc_util.broadcast_check(original_shape, broadcast_shape)
    axis_len = len(broadcast_shape) - len(original_shape)
    if axis_len > 0:
        axis = list(range(axis_len))
        broadcast_data, _ = sum_value(broadcast_data, axis, False)
        broadcast_shape = get_shape(broadcast_data)

    axis = []
    for i, _ in enumerate(original_shape):
        if original_shape[i] != broadcast_shape[i]:
            axis.append(i)
    res = sum_value(broadcast_data, axis, True)[0] if axis else broadcast_data
    return res
