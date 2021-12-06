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

"""operator dsl function: assign_add"""
import akg
import akg.topi
import akg.tvm
import akg.utils as  utils
from akg.tvm.hybrid import script
from akg.utils.dsl_create import TensorUtils

@script
def AssignAddHybrid1d(output, value, target=utils.CCE):
    """Implements assign_add ( A = A + B)."""
    for i in range(output.shape[0]):
        output[i] = output[i] + value[i]

    return output


@script
def AssignAddHybrid2d(output, value, target=utils.CCE):
    """Implements assign_add ( A = A + B)."""
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = output[i, j] + value[i, j]

    return output


@script
def AssignAddHybrid3d(output, value, target=utils.CCE):
    """Implements assign_add ( A = A + B)."""
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                output[i, j, k] = output[i, j, k] + value[i, j, k]

    return output


@script
def AssignAddHybrid4d(output, value, target=utils.CCE):
    """Implements assign_add ( A = A + B)."""
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                for l in range(output.shape[3]):
                    output[i, j, k, l] = output[i, j, k, l] + value[i, j, k, l]

    return output


@script
def AssignAddHybrid5d(output, value, target=utils.CCE):
    """Implements assign_add ( A = A + B)."""
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                for l in range(output.shape[3]):
                    for m in range(output.shape[4]):
                        output[i, j, k, l, m] = output[i, j, k, l, m] + value[i, j, k, l, m]

    return output


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor, (str, type(None)))
def AssignAdd(data, value, target=utils.CCE):
    """
    Computes data + value elementwise.

    Note:
        Only supports broadcast on input tensor value.

    Args:
        data (tvm.tensor.Tensor): Data tensor.
        value (tvm.tensor.Tensor): Value tensor, broadcast is allowed.

    Returns:
        fake_output: Invalid value, just to suit for framework.
        res: assign add result, tvm.tensor.Tensor, with same type and shape as input tensor data.
        attrs: dict.
    """
    input_shape = [x.value for x in data.shape]
    value_shape = [x.value for x in value.shape]

    if len(input_shape) < len(value_shape):
        raise RuntimeError("Do not support broadcast on input tensor data!")

    for i in range(len(value_shape)):
        if input_shape[len(input_shape) - i - 1] < value_shape[len(value_shape) - i - 1]:
            raise RuntimeError("Only support on input tensor value!")

    # broadcast adds extra compute and stage, avoid by checking the shapes before hand
    if len(value_shape) < len(input_shape) or value_shape != input_shape:
        broadcasted_value = akg.topi.broadcast_to(value, input_shape)
        res = akg.lang.ascend.vadd(data, broadcasted_value)
    else:
        res = akg.lang.ascend.vadd(data, value)
    res, binds_info = TensorUtils.inplace_set(data, res)
    attrs = {utils.BINDS: binds_info}
    return res, attrs
