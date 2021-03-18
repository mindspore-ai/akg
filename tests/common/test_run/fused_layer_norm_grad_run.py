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

from functools import reduce
from tests.common.tensorio import compare_tensor
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op import fused_layer_norm_grad as op
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def fused_layer_norm_grad_execute(shape, begin_norm_axis, begin_params_axis, dtype, attrs):
    in_rank = len(shape)
    if begin_norm_axis < 0:
        begin_norm_axis = in_rank + begin_norm_axis

    if begin_params_axis < 0:
        begin_params_axis = in_rank + begin_params_axis

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = fused_layer_norm_grad_compile(shape, begin_norm_axis, begin_params_axis, dtype, attrs, kernel_name, t)

        if t:
            # input, expect = genData(shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype)
            dbeta, dgamma, dx, dy, gamma, mean, out_dbeta, out_dgamma, out_dx, variance, x = gen_data(begin_norm_axis,
                                                                                                      begin_params_axis,
                                                                                                      dtype,
                                                                                                      in_rank, shape)
            return mod, (dx, dgamma, dbeta), {"args": (x, dy, variance, mean, gamma, out_dx, out_dgamma, out_dbeta),
                                              'outputs': (-3, -2, -1),
                                              'tuning': False}
        else:
            print('------------gen mod 1')
            return mod

    dbeta, dgamma, dx, dy, gamma, mean, out_dbeta, out_dgamma, out_dx, variance, x = gen_data(begin_norm_axis,
                                                                                              begin_params_axis, dtype,
                                                                                              in_rank, shape)

    mod = fused_layer_norm_grad_compile(shape, begin_norm_axis, begin_params_axis, dtype, attrs)
    out = utils.mod_launch(mod, (x, dy, variance, mean, gamma, out_dx, out_dgamma, out_dbeta), outputs=(-3, -2, -1),
                           expect=(dx, dgamma, dbeta))
    expect = (dx, dgamma, dbeta)

    rtol, atol = get_rtol_atol("layer_norm_grad", dtype)
    return (x, dy, variance, mean, gamma), out, expect, all(
        map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), out, expect))


def gen_data(begin_norm_axis, begin_params_axis, dtype, in_rank, shape):
    x = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    dy = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    v_m_shape = list(shape[:begin_norm_axis]) + [1]
    variance = np.broadcast_to(np.fabs(random_gaussian(v_m_shape, miu=1, sigma=0.3).astype(dtype)), shape)
    mean = np.broadcast_to(random_gaussian(v_m_shape, miu=1, sigma=0.3).astype(dtype), shape)
    gamma = random_gaussian(shape[begin_params_axis:], miu=1, sigma=0.3).astype(dtype)
    norm_axes = tuple(range(begin_norm_axis, in_rank))
    sum_num = np.array(reduce(lambda x, y: x * y, shape[begin_norm_axis:])).astype(dtype)
    param_axes = tuple(range(0, begin_params_axis))
    # avoid fp16 overflow in numpy, which can cause expect data containing -inf
    if dtype == "float16":
        np_process_type = np.float32
    else:
        np_process_type = dtype
    x_np = x.astype(np_process_type)
    dy_np = dy.astype(np_process_type)
    variance_np = variance.astype(np_process_type)
    mean_np = mean.astype(np_process_type)
    gamma_np = gamma.astype(np_process_type)
    const_two = np.array(2.0).astype(np_process_type)
    dx_ = dy_np * gamma_np
    var = 1.0 / np.sqrt(variance_np + 1e-5)
    x_mean = (x_np - mean_np)
    dva = dx_ * x_mean * (-0.5) * np.power(var, 3)
    dvariance = np.broadcast_to(np.sum(dva, axis=norm_axes, keepdims=True), shape)
    dmean_1 = np.broadcast_to(np.sum(dx_ * (-1.0) * var, axis=norm_axes, keepdims=True), shape)
    dmean_2 = dvariance * np.broadcast_to(np.mean(-2.0 * x_mean, axis=norm_axes, keepdims=True), shape)
    dmean = dmean_1 + dmean_2
    dx = dx_ * var + dvariance * const_two * x_mean / sum_num + dmean / sum_num
    dgamma = np.sum(dy_np * x_mean * var, axis=param_axes)
    dbeta = np.sum(dy_np, axis=param_axes)
    # cast back to original dtype after calculation
    dx = dx.astype(dtype)
    dgamma = dgamma.astype(dtype)
    dbeta = dbeta.astype(dtype)
    out_dx = np.full(shape, np.nan, dtype)
    out_dgamma = np.full(gamma.shape, np.nan, dtype)
    out_dbeta = np.full(gamma.shape, np.nan, dtype)
    return dbeta, dgamma, dx, dy, gamma, mean, out_dbeta, out_dgamma, out_dx, variance, x


def fused_layer_norm_grad_compile(shape, begin_norm_axis, begin_params_axis, dtype, attrs, kernel_name='fused_layer_norm_grad', tuning=False):
    x = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    dy = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    v_m_shape = list(shape[:begin_norm_axis]) + [1]
    variance = np.broadcast_to(np.fabs(random_gaussian(v_m_shape, miu=1, sigma=0.3).astype(dtype)), shape)
    mean = np.broadcast_to(random_gaussian(v_m_shape, miu=1, sigma=0.3).astype(dtype), shape)
    gamma = random_gaussian(shape[begin_params_axis:], miu=1, sigma=0.3).astype(dtype)

    return utils.op_build_test(op.fused_layer_norm_grad, [x.shape, dy.shape, variance.shape, mean.shape, gamma.shape],
                               [dtype, dtype, dtype, dtype, dtype], op_attrs=[begin_norm_axis, begin_params_axis],
                               kernel_name=kernel_name, attrs=attrs, tuning=tuning)
