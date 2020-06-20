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

"""select_run"""
import numpy as np

from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import select
from base import get_rtol_atol
from gen_random import random_gaussian

def select_run(shape_cond, shape_x, dtype_cond, dtype_x, attrs=None):
    """select_run implementation"""
    if attrs is None:
        attrs = {}

    mod = utils.op_build_test(select.select, [shape_cond, shape_x, shape_x], [dtype_cond, dtype_x, dtype_x],
                              kernel_name='select', op_attrs=[], attrs=attrs)
    args, exp_output, cond, x1, x2 = gen_data(shape_cond, shape_x, dtype_cond, dtype_x)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("select", dtype_x)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return [cond, x1, x2], acu_output, exp_output, testcase_result


def gen_data(shape_cond, shape_x, dtype_cond, dtype_x):
    # generate data
    cond = np.random.randint(0, 2, shape_cond).astype(dtype_cond)
    x1 = random_gaussian(shape_x, miu=10, sigma=0.3).astype(dtype_x)
    x2 = random_gaussian(shape_x, miu=10, sigma=0.3).astype(dtype_x)
    exp_output = np.where(cond, x1, x2)

    # inputs and output to hold the data
    output = np.full(shape_x, np.nan, dtype_x)
    args = [cond, x1, x2, output]
    return args, exp_output, cond, x1, x2
