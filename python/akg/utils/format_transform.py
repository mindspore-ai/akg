#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""format transform function"""
import akg

supported_bits = {
    "8": 1, "16": 2, "32": 4, "64": 8, "bool": 1
}


def to_tvm_const(x):
    """Convert integer to TVM expression"""
    if isinstance(x, int):
        return akg.tvm.const(x)

    return x


def get_const(expr):
    """
    get const value from TVM expression.

    Args:
        expr (tvm.expr.Expr): tvm expression.

    Returns:
        value (int): expr value.
    """
    if isinstance(expr, int):
        return expr

    if not isinstance(expr, (akg.tvm.expr.IntImm, akg.tvm.expr.UIntImm)):
        expr = akg.tvm.ir_pass.Simplify(expr)

    if not isinstance(expr, (akg.tvm.expr.IntImm, akg.tvm.expr.UIntImm)):
        raise TypeError("Expr is not a const. Get const fail, please use get shape.")
    return expr.value


def get_bytes(dtype, allow_none=False):
    """get number of bytes for supported dtype."""
    dtype = str(dtype)
    for bits in supported_bits:
        if bits in dtype:
            return supported_bits[bits]
    if allow_none:
        return None
    raise RuntimeError("Invalid dtype, supported bits are {0}".format(supported_bits.keys()))


def refine_shape(shape, reduce_axis=None):
    """
    Refine shape to drop 1 in shape according to reduce axis.

    Note:
        if input is just shape, result is shape, and if inputs are shape and axis, result is a tuple of (shape, axis).

    Args:
        shape : shape of data
        reduce_axis : list, tuple or int
            axis want to reduce

    Returns:
        shape (list): refined shape.
        reduce_axis (list): if input parameters send reduce axis, this will be the output.
        if all the reduce axis is illegal like the length of reduce axis is 1, a empty list([]) will be returned.
    """

    def _refine_shape_no_reduce():
        refined = [shp for _, shp in enumerate(shape) if shp > 1]
        if not refined:
            refined = [1]
        return refined

    def _update_res_reduce_axis(res_reduce_axis_list, cnt):
        for j, axs in enumerate(res_reduce_axis_list):
            if axs > cnt:
                res_reduce_axis_list[j] -= 1

    if reduce_axis is not None:
        res_reduce_axis = sorted(refine_reduce_axis(shape, reduce_axis))
        if not res_reduce_axis:
            return _refine_shape_no_reduce(), []
        res_shape = shape[:]
        refined_shape = []
        count = 0
        for i in res_shape:
            if i > 1:
                refined_shape.append(i)
                count += 1
            else:
                _update_res_reduce_axis(res_reduce_axis, count)

        return refined_shape, res_reduce_axis
    return _refine_shape_no_reduce()


def refine_reduce_axis(input_shape, axis):
    """make reduce axis legal."""
    shape = get_shape(input_shape)
    if axis is None:
        axis = [i for i in range(len(shape))]
    elif isinstance(axis, int):
        axis = [axis]
    elif not isinstance(axis, (tuple, list)):
        raise TypeError("axis must be one of the type int,tuple,list or None")

    if len(axis) > len(shape):
        raise ValueError("axis size must not larger than shape size")

    axis = list(axis)

    for i, _ in enumerate(axis):
        if axis[i] < 0:
            axis[i] += len(shape)

        if axis[i] >= len(shape):
            raise ValueError(("axis value-{} exceeds len(axis) which is invalid".format(axis[i])))

    axis.sort(reverse=True)

    return axis


def get_shape_from_tensor(data):
    """translate akg.tvm.shape to list type in python."""
    tvm_shape = data.shape
    py_shape = []
    for i in tvm_shape:
        if isinstance(i, akg.tvm.expr.IntImm):
            py_shape.append(i.value)
        else:
            py_shape.append(i)
    return py_shape


def tvm_shape_to_list(tvm_shape):
    """translate akg.tvm.shape to list type in python."""
    py_shape = []
    for i in tvm_shape:
        if isinstance(i, akg.tvm.expr.Var):
            py_shape.append(i)
        else:
            py_shape.append(i.value)
    return py_shape


def tvm_array_to_list(tvm_array):
    """translate akg.tvm.array to list type in python."""
    tensor_list = []
    for i in tvm_array:
        if isinstance(i, akg.tvm.tensor.Tensor):
            tensor_list.append(i)
        else:
            raise ValueError("Only surpport akg.tvm.tensor.Tensor.")
    return tensor_list


def get_shape(data):
    """get shape and save it as list."""
    if isinstance(data, akg.tvm.tensor.Tensor):
        shape = get_shape_from_tensor(data)
    elif isinstance(data, akg.tvm.container.Array):
        shape = tvm_shape_to_list(data)
    elif isinstance(data, int):
        shape = [data]
    elif isinstance(data, (tuple, list)):
        shape = list(data)
    elif isinstance(data, akg.tvm.expr.Var):
        shape = [data]
    else:
        raise TypeError("Refine axis does not support type {} for now.".format(type(data)))
    return shape


def convert_to_list(something, convert_all=True):
    """convert other types to string."""
    out = []
    if isinstance(something, (list, tuple)):
        for x in something:
            out.append(convert_to_list(x, convert_all=False))
    else:
        if convert_all:
            out.append(something)
        else:
            out = something
    return out


def to_tvm_nd_array(data, ctx=None):
    """convert other types to tvm nd array with specified context"""
    if ctx is None:
        ctx = akg.tvm.context("cuda", 0)
    if isinstance(data, (list, tuple)):
        return [akg.tvm.nd.array(d, ctx) for d in data]
    return akg.tvm.nd.array(data, ctx)
