#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: mean"""
import akg.topi
import akg.tvm
from akg.utils import format_transform as ft_util
import akg.utils as utils
from akg.utils import custom_tiling as ct_util
from akg.utils.dynamic_shape import shape_is_dynamic
from ..sum import sum
from .sum_others import sum_v2


def get_attrs(tensor):
    """generate default attrs."""
    if shape_is_dynamic(tensor):
        return {"enable_double_buffer": 0, "enable_divide_var": 1}

    return {}


def mean_dynamic_tiling_strategy(tensor, axis):
    """custom tiling for mean with dynamic shape"""
    strategy = list()

    inner_most_to_full = True
    resnet_inner_most_axis_pos = 4

    reduce_axis_to_1 = True
    reduce_axis_to_no_iso = False

    multicore_axis_to_1 = True
    resnet_outer_most_axis_pos = 0

    if inner_most_to_full:
        strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                        values="FULL",
                                                        constraints=ct_util.TileConstraint.MAX,
                                                        tensor_pos=resnet_inner_most_axis_pos)
    if reduce_axis_to_1:
        strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                        values=[1 for _ in axis],
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=axis)
    elif reduce_axis_to_no_iso:
        strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                        values=[1 for _ in axis],
                                                        constraints=ct_util.TileConstraint.FORBID_ISOLATE,
                                                        tensor_pos=axis)
    if multicore_axis_to_1:
        strategy += ct_util.create_constraint_on_tensor(tensor=tensor,
                                                        values=1,
                                                        constraints=ct_util.TileConstraint.FACTOR,
                                                        tensor_pos=resnet_outer_most_axis_pos)
    return strategy


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple, int, type(None)), (bool, type(None)), (str, type(None)))
def mean(data, axis=None, keepdims=False, target=utils.CCE):
    """
    Computes the mean of the values of a Tensor over the whole dataset.

    Note:
        If the tuple's elements are unsorted, this function will call preprocess_axis firstly to let these elements
        sorted. if tuple is empty, this function will compute all elements' sum.
        if the data type is folat 16 and the whole dim not less than 65536, this function will compute the mean by
        divide 65535 first to avoid whole dim too large.

    Args:
        data (tvm.tensor.Tensor): Tensor of type float16, float32.
        axis (Union[list, tuple, int, None]): If the tuple is empty, the axis equal to None.
        keepdims (bool): If keepdims equal to True, the result shape length is same to input shape length.

    Returns:
            tvm.tensor.Tensor, has the same type as data. If keepdims equal to True, all reduced dimensions are
            retained with length 1. else these reduced axis will be eliminate.

    Supported Platforms:
        'Ascend'
    """
    # Check types
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)

    # Check shape
    shape = ft_util.get_shape(data)
    utils.reduce_axis_check(shape, axis)
    axis = ft_util.refine_reduce_axis(data, axis)

    count = 1
    for i in axis:
        count *= shape[i]
    output = sum(data, axis, keepdims, target=target)

    if shape_is_dynamic(data):
        res = akg.tvm.compute(output.shape, lambda *i: akg.lang.ascend.divide_var(output(*i), count), name="res")
    else:
        res = akg.topi.divide(output, count)

    attrs = get_attrs(data)
    if shape_is_dynamic(data):
        attrs["custom_tiling"] = mean_dynamic_tiling_strategy(data, axis)
    return res, attrs


@utils.check_input_type(akg.tvm.tensor.Tensor, (list, tuple, int, type(None)), (bool, type(None)), (str, type(None)))
def mean_v2(data, axis=None, keepdims=False, target=utils.CCE):
    """
    Simple implementation of mean.

    Supported Platforms:
        'Ascend'
    """
    # Check types
    utils.ops_dtype_check(data.dtype, utils.DtypeForDavinci.ALL_FLOAT)

    # Check shape
    shape = [x.value for x in data.shape]
    utils.reduce_axis_check(shape, axis)
    axis = ft_util.refine_reduce_axis(data, axis)

    dtype = data.dtype
    count = 1
    for i in axis:
        count *= shape[i]

    count_rec = 1 / count
    output = sum_v2(data, axis, keepdims, target=target)
    res = output * akg.tvm.const(count_rec, dtype)
    attrs = get_attrs(data)
    if shape_is_dynamic(data):
        attrs["custom_tiling"] = mean_dynamic_tiling_strategy(data, axis)
    return res, attrs
