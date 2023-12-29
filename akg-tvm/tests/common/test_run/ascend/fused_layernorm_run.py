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
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import fused_layernorm as op
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def fused_layernorm_execute(shape_x, begin_norm_axis, begin_params_axis, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = fused_layernorm_compile(shape_x, begin_norm_axis, begin_params_axis, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            beta, expect, gamma, input, out_mean, out_variance, output = gen_data(begin_norm_axis, begin_params_axis,
                                                                                  dtype, shape_x)
            return mod, expect, {"args": (input, gamma, beta, output, out_mean, out_variance), 'outputs': (-3, -2, -1),
                                 'tuning': False}
        else:
            return mod
    else:
        mod = fused_layernorm_compile(shape_x, begin_norm_axis, begin_params_axis, dtype, attrs)
        beta, expect, gamma, input, out_mean, out_variance, output = gen_data(begin_norm_axis, begin_params_axis, dtype,
                                                                              shape_x)
        res = utils.mod_launch(mod, (input, gamma, beta, output, out_mean, out_variance), outputs=(-3, -2, -1),
                               expect=expect)
        rtol, atol = get_rtol_atol("layernorm", dtype)
        return (input, gamma, beta, begin_norm_axis, begin_params_axis), res, expect, \
            all(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), res, expect))


def gen_data(begin_norm_axis, begin_params_axis, dtype, shape_x):
    input = random_gaussian(shape_x, miu=1, sigma=0.1).astype(dtype)
    gamma = random_gaussian(shape_x[begin_params_axis:], miu=1, sigma=0.1).astype(dtype)
    beta = random_gaussian(shape_x[begin_params_axis:], miu=1, sigma=0.1).astype(dtype)
    in_rank = len(shape_x)
    if begin_norm_axis < 0:
        norm_axis = begin_norm_axis + in_rank
    else:
        norm_axis = begin_norm_axis
    norm_axes = tuple(range(norm_axis, in_rank))
    mean = np.broadcast_to(np.mean(input, axis=norm_axes, keepdims=True), shape_x)
    diff = input - mean
    square = np.square(diff)
    smean = np.broadcast_to(np.mean(square, axis=norm_axes, keepdims=True), shape_x)
    meps = smean + 1e-5
    # sqrt = np.sqrt(meps)
    # rsqrt = 1.0 / sqrt
    logs = np.log(meps)
    mul = logs * (-0.5)
    rsqrt = np.exp(mul)
    out = diff * rsqrt
    bn = out * gamma + beta
    output = np.full(shape_x, np.nan, dtype)
    out_mean = np.full(shape_x, np.nan, dtype)
    out_variance = np.full(shape_x, np.nan, dtype)
    expect = (bn, mean, smean)
    return beta, expect, gamma, input, out_mean, out_variance, output


def fused_layernorm_compile(shape_x, begin_norm_axis, begin_params_axis, dtype, attrs, kernel_name="fused_layernorm", tuning=False):
    input = random_gaussian(shape_x, miu=1, sigma=0.1).astype(dtype)
    gamma = random_gaussian(shape_x[begin_params_axis:], miu=1, sigma=0.1).astype(dtype)
    beta = random_gaussian(shape_x[begin_params_axis:], miu=1, sigma=0.1).astype(dtype)
    return utils.op_build_test(op.fused_layernorm, [input.shape, gamma.shape, beta.shape], [dtype, dtype, dtype],
                               op_attrs=[begin_norm_axis, begin_params_axis], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
