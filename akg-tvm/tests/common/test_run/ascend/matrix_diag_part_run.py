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

"""matrix_diag_part_run"""
import numpy as np

from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import matrix_diag_part
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol

def matrix_diag_part_run(shape, dtype, attrs=None):
    """matrix_diag_part_run"""
    mod = utils.op_build_test(matrix_diag_part.matrix_diag_part, [shape, shape], [dtype, dtype],
                              kernel_name='matrix_diag_part', attrs=attrs)
    args, exp_output, input_x, input_help = gen_data(shape, dtype)
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("matrix_diag_part", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)
    return [input_x, input_help], acu_output, exp_output, testcase_result


def gen_data(shape, dtype):
    """generate valid data to test"""
    input_x = random_gaussian(shape, miu=10, sigma=0.3).astype(dtype)
    # need to generate a multi dimensional 01 diagonal matrix by broadcasting a 2D 01 diagonal matrix
    input_help = np.zeros((shape[-2], shape[-1]))
    for i in range(min(shape[-2], shape[-1])):
        input_help[i, i] = 1.0
    input_help = np.broadcast_to(input_help, shape)
    input_help = input_help.astype(dtype)

    exp_output = input_x * input_help
    if input_x.shape[-2] < input_x.shape[-1]:
        exp_output = np.sum(exp_output, -1)
    else:
        exp_output = np.sum(exp_output, -2)
    # inputs and output to hold the data
    output = np.full(exp_output.shape, np.nan, dtype)
    args = [input_x, input_help, output]
    return args, exp_output, input_x, input_help
