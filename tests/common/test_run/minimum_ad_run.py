# Copyright 2020 Huawei Technologies Co., Ltd
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

"""minimum_ad_run"""
import numpy as np

from akg.utils import kernel_exec as utils
from test_op import minimum_ad
from tensorio import compare_tensor
from base import get_rtol_atol

def minimum_ad_run(shape, dtype, grad_x=True, grad_y=True, attrs=None):
    """minimum_ad_run implementation"""
    mod = utils.op_build_test(minimum_ad.minimum_ad, [shape, shape, shape], [dtype, dtype, dtype],
                              kernel_name='minimum_ad', op_attrs=[grad_x, grad_y], attrs=attrs)
    rtol, atol = get_rtol_atol("minimum_ad", dtype)
    grads, data_x, data_y, output_dx, output_dy, exp_dx, exp_dy = gen_data(shape, dtype)
    inputs = (grads, data_x, data_y)
    if grad_x and grad_y:
        args = (*inputs, output_dx, output_dy)
        expects = (exp_dx, exp_dy)
        outputs = utils.mod_launch(mod, args, outputs=(-2, -1), expect=expects)

    elif grad_x:
        args = (*inputs, output_dx)
        expects = exp_dx
        outputs = utils.mod_launch(mod, args, outputs=(-1,), expect=expects)

    else:
        args = (*inputs, output_dy)
        expects = exp_dy
        outputs = utils.mod_launch(mod, args, outputs=(-1,), expect=expects)

    testcase_result = compare_tensor(outputs, expects, rtol=rtol, atol=atol, equal_nan=True)
    return inputs, outputs, expects, testcase_result

def gen_data(shape, dtype):
    """generate valid data to test"""
    if dtype == 'int32':
        low_bound = -1000
        high_bound = 1000
    else:
        low_bound = -1.0
        high_bound = 1.0

    data_x = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(dtype)
    data_y = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(dtype)
    grads = np.random.uniform(low=low_bound, high=high_bound, size=tuple(shape)).astype(dtype)

    # If data_y >= data_x : expect_dx = data_dz, expect_dy = 0; else expect_dx = 0, expect_dy = data_dz.
    # So, if data_y >= data_x, incidator_x = 1, else incidator_x = 0.
    incidator_x = np.zeros(data_x.shape, dtype)
    incidator_x[np.where(data_x <= data_y)] = 1
    incidator_x.astype(dtype)
    incidator_y = np.abs((incidator_x - 1).astype(dtype))

    expect_dx = grads * incidator_x
    expect_dy = grads * incidator_y
    # inputs and output to hold the data
    output_dx = np.full(shape, np.nan, dtype)
    output_dy = np.full(shape, np.nan, dtype)
    return grads, data_x, data_y, output_dx, output_dy, expect_dx, expect_dy

