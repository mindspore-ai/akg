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

"""broadcat compute"""

import akg.tvm
from .util import dtype_check_decorator, shape_to_list, judge_var

_name_index = [0]


@dtype_check_decorator
def broadcast(var, shape, output_dtype=None):
    """
    broadcast scalar to tensor, only support float16

    Args:
        var (Union[int, float, tvm.const]): input
        shape (tvm.tensor.Tensor): shape
        output_dtype (tvm.tensor.Tensor): var.dtype

    Returns:
        tvm.tensor.Tensor, broadcast tensor
    """
    if isinstance(shape, akg.tvm.container.Array):
        shape = shape_to_list(shape)
    if isinstance(var, akg.tvm.tensor.Tensor):
        tensor = var
        orig_shape = shape_to_list(tensor.shape)
        if len(orig_shape) > len(shape):
            raise RuntimeError(
                "Length of shape of input must be less than or equal to output for Tensor Broadcasting, while " +
                "input shape is %s, and output shape is %s" % (str(orig_shape), str(shape)))
        expand_shape_len = len(shape) - len(orig_shape)
        check_equal = 0
        for so, sd in zip(orig_shape, shape[expand_shape_len:]):
            if so == sd:
                check_equal += 1
                continue
            elif so == 1:
                continue
            raise RuntimeError(
                "For tensor broadcasting, shape must be the same or corresponding shape of src tensor is 1"
                "while src shape is %s, and dst shape is %s" % (str(orig_shape), str(shape)))
        if check_equal == len(shape):
            return tensor

        name = "broadcast_tensor_" + str(_name_index[0])
        _name_index[0] += 1

        op = 'broadcast_for_tensor'
        lambda_func = lambda *indice: tensor(*([0 if orig_shape[i] == 1
                                                else indice[i + expand_shape_len] for i in range(len(orig_shape))]))

        with akg.tvm.tag_scope(op):
            out = akg.tvm.compute(shape, lambda_func, name=name)
        return out
    var_type = judge_var(var)
    tmp_args = var
    if var_type == "python_const":
        if isinstance(tmp_args, float):
            tmp_args = akg.tvm.const(tmp_args, dtype="float16")
        else:
            tmp_args = akg.tvm.const(tmp_args, dtype="int32")

    if not output_dtype:
        output_dtype = tmp_args.dtype

    tmp_args = tmp_args.astype(output_dtype)

    lambda_func = lambda *indice: tmp_args

    name = "broadcast_" + str(_name_index[0])
    _name_index[0] += 1

    op = 'broadcast'
    with akg.tvm.tag_scope(op):
        out = akg.tvm.compute(shape, lambda_func, name=name)
    return out
