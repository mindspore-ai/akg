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

"""operator dsl function: reduce_min_max_common"""
import akg.topi
import akg.tvm
from akg.utils import validation_check as vc_util
from akg.utils.format_transform import refine_reduce_axis
from akg.ops.math.cast import cast


@vc_util.check_input_type(akg.tvm.tensor.Tensor, (int, list, tuple, type(None)),
                          (bool, type(None)), (str, type(None)))
def reduce_min_max(data, axis=None, keepdims=False, method="min"):
    """
    Computes the maximum or minimum of elements over a given axis or a list of axes of a tensor.

    Args:
        data (tvm.tensor.Tensor): The input tensor to reduce. Should be of type float16, float32, int8, uint8, int32.
        axis (Union[list, tuple, int, None]): The dimensions to reduce.
                                      If None, all dimensions will be reduced.
                                      If int or list, must be in the range [-len(data.shape), len(data.shape) - 1].
        keepdims (bool): If True, retains reduced dimensions with length 1, default value is False.
        method (str): Specifies to compute maximum or minimum of input tensor, default value is min.

    Returns:
        tvm.tensor.Tensor of same type as input tensor data.
    """
    # check shape
    vc_util.check_shape(data.shape)

    # check type
    dtype = data.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_TYPES)

    # check axis
    shape_len = len(data.shape)
    if axis is None:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)
    if isinstance(axis, int):
        axis = [axis]
    vc_util.is_valid_reduce_axis(data, axis)
    refined_axis = refine_reduce_axis(data, axis)
    if len(set(refined_axis)) == len(data.shape) and not keepdims:
        raise ValueError("When reducing on all axes of input, keepdim should be set to True.")
    # check method
    method_list = ["min", "max"]
    if method not in method_list:
        raise ValueError("supported method %s while given method is %s" % (",".join(method_list), method))

    # In the emit_insn pass, for vmin and vmax, reduce_last_axis only support float16.
    if dtype != "float16":
        data = cast(data, "float16")

    if method == "min":
        res = akg.topi.min(data, axis=axis, keepdims=keepdims)
    else:
        res = akg.topi.max(data, axis=axis, keepdims=keepdims)

    if res.dtype != dtype:
        res = cast(res, dtype)

    return res
