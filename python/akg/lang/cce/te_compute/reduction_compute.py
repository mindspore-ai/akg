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

"""reduction compute"""
from decorator import decorator
import akg.tvm
from .cast_compute import cast
from .util import save_op_output_dtype, shape_to_list, refine_axis


reduce_supported_types = {
    "sum": ["float16", "float32"],
    "reduce_min": ["float16"],
    "reduce_max": ["float16"],
}


@decorator
def auto_cast_of_reduce(func, *args, **kwargs):
    """
    auto cast dectorator.

    Note:
        Before calling elewise api, check the input tensor is supported by the intr.
        If not supported, casting the input tensor to supported dtype. (On condition
        that the cast type is supported.If the cast type is not supported,raising
        a RuntimeError).
    """
    intr = func.__name__

    save_op_output_dtype(func, *args)

    supported_types = reduce_supported_types[intr]

    if len(args) == 3:
        raw_tensor = args[0]
        axis = args[1]
        keepdims = args[2]

        dtype = raw_tensor.dtype

        temp_tensor = raw_tensor
        if dtype not in supported_types:
            temp_tensor = cast(raw_tensor, "float16")

        return func(temp_tensor, axis, keepdims)
    return func(*args, **kwargs)


name_index = [0]


@auto_cast_of_reduce
def sum(raw_tensor, axis, keepdims=False):
    """
    calculate sum of raw_tensor, only support float16

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor
        axis (Union[int, list]): reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
        keepdims (bool): if true, retains reduced dimensions with length 1, default value is None

    Returns:
        tvm.tensor.Tensor, res
    """
    return single_reduce_op(raw_tensor, axis, "reduce_sum", keepdims)


@auto_cast_of_reduce
def reduce_min(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_min of raw_tensor, only support float16

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor
        axis (Union[int, list]): reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])
        keepdims (bool): if true, retains reduced dimensions with length 1, default value is None

    Returns:
        tvm.tensor.Tensor, res
    """
    return single_reduce_op(raw_tensor, axis, "reduce_min", keepdims)


@auto_cast_of_reduce
def reduce_max(raw_tensor, axis, keepdims=False):
    """
    calculate reduce_max of raw_tensor, only support float16

    Args:
        raw_tensor (tvm.tensor.Tensor): input tensor
        keepdims (bool): if true, retains reduced dimensions with length 1, default value is None
        axis (Union[int, list]): reduce axis (range : [-len(raw_tensor.shape), len(raw_tensor.shape) - 1])

    Returns:
        tvm.tensor.Tensor, res
    """
    return single_reduce_op(raw_tensor, axis, "reduce_max", keepdims)


def single_reduce_op(input_tensor, axis, op, keepdims=False):
    """factory method of single reduce operations"""
    def reduce_compute(data_shape, axis, tensor, func):
        def compute_func(*indice):
            count_indice = 0
            count_reduce = 0
            res_indice = []
            for index in range(len(data_shape)):
                if index not in axis:
                    res_indice.append(indice[count_indice])
                    count_indice += 1
                else:
                    res_indice.append(reduce_axises[count_reduce])
                    count_reduce += 1
                    if keepdims:
                        count_indice += 1

            return func(tensor(*res_indice), axis=reduce_axises)

        reduce_axises = []
        for index, axis_num in enumerate(axis):
            reduce_axises.append(akg.tvm.reduce_axis((0, data_shape[axis_num]), name='k' + str(index + 1)))
        res_reshape = []
        for index, shape_l in enumerate(data_shape):
            if index not in axis:
                res_reshape.append(shape_l)
            else:
                if keepdims:
                    res_reshape.append(1)
        if is_last_axis and not keepdims:
            res_reshape.append(1)

        name = "reduce_" + str(name_index[0])
        name_index[0] += 1

        reduce_res = akg.tvm.compute(res_reshape, compute_func, name=name)
        return reduce_res

    if op.lower() == "reduce_min":
        reduce_func = akg.tvm.min
    elif op.lower() == "reduce_max":
        reduce_func = akg.tvm.max
    elif op.lower() == "reduce_sum":
        reduce_func = akg.tvm.sum
    else:
        raise RuntimeError("Not Support yet for op %s." % op)

    op_tensor = input_tensor
    shape = shape_to_list(op_tensor.shape)
    res_axis = refine_axis(axis, shape)

    if not res_axis:
        return input_tensor

    for i in res_axis:
        is_last_axis = (i == len(shape) - 1)
        if is_last_axis:
            break

    with akg.tvm.tag_scope(op.lower()):
        res = reduce_compute(shape, res_axis, op_tensor, reduce_func)

    return res
