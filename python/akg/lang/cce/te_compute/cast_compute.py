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

"""cast compute"""
from decorator import decorator
import akg.tvm
from .util import save_op_output_dtype, get_intr_types, is_cast_support, shape_to_list

name_index = [0]


@decorator
def auto_cast_of_cast(func, *args, **kwargs):
    """
    auto cast dectorator.

    Note:
        Before calling elewise api, check the input tensor is supported by the intr.
        If not supported, casting the input tensor to supported dtype.

    Raises:
        TypeError, If the cast type is not supported.
    """
    intr = func.__name__

    save_op_output_dtype(func, *args)

    supported_types = get_intr_types("Intrinsic_" + intr)
    if len(args) == 1:
        raw_tensor = args[0]
        src_dtype = raw_tensor.dtype

        temp_tensor = raw_tensor
        if src_dtype not in supported_types:
            if "float32" in supported_types and is_cast_support(src_dtype, "float32"):
                temp_tensor = cast(raw_tensor, "float32")
            else:
                temp_tensor = cast(raw_tensor, "float16")
        return func(temp_tensor)
    return func(*args, **kwargs)


def cast(raw_tensor, dst_dtype):
    """
    cast tensor from src_type to dst_dtype, only support f322f16, f162f32, f162s8, s82f16, f162u8, u82f16

    Args:
        raw_tensor (tvm.tensor.Tensor): input
        dst_dtype : destinatin type

    Returns:
        tvm.tensor.Tensor, casted tensor
    """
    src_dtype = raw_tensor.dtype
    dst_dtype_lower = dst_dtype.lower()
    if dst_dtype_lower == src_dtype:
        return raw_tensor

    if not is_cast_support(src_dtype, dst_dtype_lower):
        if is_cast_support(src_dtype, "float32") and is_cast_support("float32", dst_dtype_lower):
            raw_tensor = cast_op(raw_tensor, "float32", 'elewise_single_cast')
        elif is_cast_support(src_dtype, "float16") and is_cast_support("float16", dst_dtype_lower):
            raw_tensor = cast_op(raw_tensor, "float16", 'elewise_single_cast')
        else:
            raise TypeError("Unsupported cast type!")

    return cast_op(raw_tensor, dst_dtype_lower, 'elewise_single_cast')


@auto_cast_of_cast
def ceil(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with ceiling method

    Args:
        raw_tensor (tvm.tensor.Tensor): input

    Returns:
        tvm.tensor.Tensor, casted tensor
    """
    dst_dtype = "int32"

    return cast_op(raw_tensor, dst_dtype, "elewise_single_ceil")


@auto_cast_of_cast
def floor(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with flooring method

    Args:
        raw_tensor (tvm.tensor.Tensor): input

    Returns:
        tvm.tensor.Tensor, casted tensor
    """
    dst_dtype = "int32"

    return cast_op(raw_tensor, dst_dtype, "elewise_single_floor")


@auto_cast_of_cast
def round(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with rounding method

    Args:
        raw_tensor (tvm.tensor.Tensor): input

    Returns:
        tvm.tensor.Tensor, casted tensor
    """
    dst_dtype = "int32"

    return cast_op(raw_tensor, dst_dtype, "elewise_single_round")


@auto_cast_of_cast
def trunc(raw_tensor):
    """
    cast tensor from src_type to dst_dtype with trunc method

    Args:
        raw_tensor (tvm.tensor.Tensor): input

    Returns:
        tvm.tensor.Tensor, casted tensor
    """
    dst_dtype = "int32"

    return cast_op(raw_tensor, dst_dtype, "elewise_single_trunc")


def cast_op(input_tensor, output_dtype, op):
    """factory method of single elewise operations"""
    in_tensor = input_tensor
    shape = shape_to_list(in_tensor.shape)

    if op == "elewise_single_cast":
        lambda_func = lambda *indice: in_tensor(*indice).astype(output_dtype)
    elif op == "elewise_single_round":
        lambda_func = lambda *indice: akg.tvm.round(in_tensor(*indice)).astype(output_dtype)
    elif op == "elewise_single_ceil":
        lambda_func = lambda *indice: akg.tvm.ceil(in_tensor(*indice)).astype(output_dtype)
    elif op == "elewise_single_floor":
        lambda_func = lambda *indice: akg.tvm.floor(in_tensor(*indice)).astype(output_dtype)
    elif op == "elewise_single_trunc":
        lambda_func = lambda *indice: akg.tvm.trunc(in_tensor(*indice)).astype(output_dtype)
    else:
        raise ValueError("operation %s not support yet" % op)

    name = op.split("_")[-1] + "_" + str(name_index[0])
    name_index[0] += 1

    with akg.tvm.tag_scope(op):
        tmp = akg.tvm.compute(shape, lambda_func, name=name)
    return tmp
