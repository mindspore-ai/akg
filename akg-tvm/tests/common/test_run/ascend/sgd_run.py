# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""sgd_run"""
import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
from tests.common.test_op.ascend import sgd
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def sgd_run(shape, dtype, nesterov=False, dampening=0.0, weight_decay=0.0, lr_mat=0.1, momt_mat=0.9, attrs=None):
    """run function for dsl function sgd."""
    lr = np.full((1,), lr_mat).astype(dtype)
    momt = np.full((1,), momt_mat).astype(dtype)
    mod = utils.op_build_test(sgd.sgd, [shape, shape, shape, shape, lr.shape, momt.shape],
                              [dtype, dtype, dtype, dtype, dtype, dtype], [dampening, weight_decay, nesterov],
                              kernel_name='sgd', attrs=attrs)
    parameters, gradient, accum, stat, parameters_t, accum_t, stat_t, output_para, output_accum, output_stat \
        = gen_data(dtype, shape, lr, momt, dampening, weight_decay, nesterov)
    output_para, output_accum, output_stat = utils.mod_launch(mod, (parameters, gradient, accum, stat, lr, momt),
                                                              outputs=(0, 2, 3),
                                                              expect=(parameters_t, accum_t, stat_t))
    expects = (parameters_t, accum_t, stat_t)
    outputs = (output_para, output_accum, output_stat)
    rtol, atol = get_rtol_atol("sgd", dtype)
    testcase_result = compare_tensor(outputs, expects, rtol=rtol, atol=atol, equal_nan=True)

    return (parameters, gradient, accum, stat), (output_para, output_accum, output_stat), \
           (parameters_t, accum_t, stat_t), testcase_result

def gen_data(dtype, shape, lr, momt, dampening, weight_decay, nesterov):
    """Generate data for testing the op"""
    parameters = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    gradient = random_gaussian(shape, miu=3, sigma=0.3).astype(dtype)
    accum = random_gaussian(shape, miu=4, sigma=0.3).astype(dtype)
    stat = random_gaussian(shape, miu=5, sigma=0.3).astype(dtype)
    if weight_decay != 0.0:
        parameters = parameters * 1.0
        grad_delta = parameters * weight_decay
        gradient_new = gradient + grad_delta
    else:
        gradient_new = gradient
    stat_mid = -1.0 * stat
    stat_act = stat_mid + 1.0

    dampening_t = stat_act * dampening

    # update accum
    accum_delta = accum * momt[0]

    gradient_damp = gradient_new * dampening_t
    accum_t = gradient_new + accum_delta
    if dampening != 0.0:
        accum_t = accum_t - gradient_damp

    # update parameters
    if nesterov:
        parameters_delta = gradient_new * lr[0]
        parameters_delta_2 = accum_t * momt[0] * lr[0]
        parameters_delta = parameters_delta + parameters_delta_2
        parameters_t = parameters - parameters_delta
    else:
        parameters_delta = accum_t * lr[0]
        parameters_t = parameters - parameters_delta
    # update stat
    stat_t = stat_act * 0.0
    output_para = np.full(shape, np.nan, dtype)
    output_accum = np.full(shape, np.nan, dtype)
    output_stat = np.full(shape, np.nan, dtype)
    return parameters, gradient, accum, stat, parameters_t, accum_t, stat_t, output_para, output_accum, output_stat




