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

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend.fused_minimum_or_maximum_grad import fused_minimum_or_maximum_grad
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def broadcast_grad(data_broadcast, ori_shape):
    ori_dtype = data_broadcast.dtype
    if list(data_broadcast.shape) == list(ori_shape):
        return data_broadcast
    data_broadcast = data_broadcast.astype("float32")
    if list(ori_shape) == [1]:
        tmp = np.sum(data_broadcast, keepdims=True).reshape([1])
        return tmp.astype(ori_dtype)
    axis_len = len(data_broadcast.shape) - len(ori_shape)
    if axis_len > 0:
        axis = tuple(range(axis_len))
        data_broadcast = np.sum(data_broadcast, axis, keepdims=False)
    axis = []
    for i, _ in enumerate(data_broadcast.shape):
        if data_broadcast.shape[i] != ori_shape[i]:
            axis.append(i)
    res = np.sum(data_broadcast, tuple(axis), keepdims=True) if axis else data_broadcast
    return res.astype(ori_dtype)


def genData(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype):
    """ Generate data for testing the op """
    shapes = [shape_x, shape_y, shape_dz]
    inputs = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input = random_gaussian(shape, miu=1, sigma=0.1 + i * 0.1).astype(dtype)
        inputs.append(input)

    input_x = np.broadcast_to(inputs[0], shape_dz)
    input_y = np.broadcast_to(inputs[1], shape_dz)
    if op_type is "LE":
        dx = np.where(input_x <= input_y, inputs[2], 0).astype(dtype)
        dy = np.where(input_x <= input_y, 0, inputs[2]).astype(dtype)
    elif op_type is "GE":
        dx = np.where(input_x >= input_y, inputs[2], 0).astype(dtype)
        dy = np.where(input_x >= input_y, 0, inputs[2]).astype(dtype)

    dx = broadcast_grad(dx, shape_x)
    dy = broadcast_grad(dy, shape_y)

    outs = []
    if grad_x and grad_y:
        outs = [dx, dy]
    elif grad_x and grad_y is False:
        outs = dx
    elif grad_y and grad_x is False:
        outs = dy

    return inputs, outs


def fused_minimum_or_maximum_grad_execute(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = fused_minimum_or_maximum_grad_compile(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype,
                                                    kernel_name, attrs, t)
        if t:
            input, expect = genData(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype)
            output_dx = np.full(expect[0].shape, 0, dtype)
            output_dy = np.full(expect[1].shape, 0, dtype)
            return mod, expect, {"args": (input[2], input[0], input[1], output_dx, output_dy), 'outputs': (-2, -1),
                                 'tuning': False}
        else:
            return mod
    input, expect = genData(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype)
    mod = fused_minimum_or_maximum_grad_compile(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype,
                                                kernel_name, attrs)

    # get the result from mod
    rtol, atol = get_rtol_atol("fused_minimum_or_maximum_grad", dtype)
    if grad_x and grad_y:
        output_dx = np.full(expect[0].shape, 0, dtype)
        output_dy = np.full(expect[1].shape, 0, dtype)
        res = utils.mod_launch(mod, (input[2], input[0], input[1], output_dx, output_dy), (-2, -1), expect=expect)
        return (input[2], input[0], input[1]), res, expect, all(
            map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol, equal_nan=True), res, expect))

    else:
        output = np.full(expect.shape, 0, dtype)
        output = utils.mod_launch(mod, (input[2], input[0], input[1], output), expect=expect)
        test_case_result = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)
        return input, output, expect, test_case_result


def fused_minimum_or_maximum_grad_compile(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype, kernel_name="", attrs={}, tuning=False):
    return utils.op_build_test(fused_minimum_or_maximum_grad, [shape_dz, shape_x, shape_y], [dtype, dtype, dtype],
                               op_attrs=[grad_x, grad_y, op_type], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
