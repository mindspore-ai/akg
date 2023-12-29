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

"""matrix_set_diag_run"""
import numpy as np

from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import matrix_set_diag
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol

def matrix_set_diag_run(shape_matrix, shape_diagonal, dtype, attrs=None):
    """matrix_set_diag_run"""
    mod = utils.op_build_test(matrix_set_diag.matrix_set_diag, [shape_matrix, shape_diagonal, shape_matrix],
                              [dtype, dtype, dtype], kernel_name='matrix_set_diag', attrs=attrs)
    args, exp_output, input_matrix, input_diagonal, input_help = gen_data(shape_matrix, shape_diagonal, dtype)

    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("matrix_set_diag", dtype)
    testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return [input_matrix, input_diagonal, input_help], acu_output, exp_output, testcase_result

def gen_data(shape_matrix, shape_diagonal, dtype):
    """generate valid data to test"""
    input_matrix = random_gaussian(shape_matrix, miu=10, sigma=0.3).astype(dtype)
    input_diagonal = random_gaussian(shape_diagonal, miu=5, sigma=0.3).astype(dtype)
    # make shape_diagonal can support broadcast
    if shape_matrix[-2] <= shape_matrix[-1]:
        shape_b_newshape = list(shape_diagonal) + [1]
    # The penultimate dimension of the shape_diag is extended for broadcast.
    else:
        shape_b_newshape = list(shape_diagonal)
        shape_b_newshape.insert(-1, 1)
    new_input_diagonal = np.reshape(input_diagonal, shape_b_newshape)
    # need to generate a multi dimensional 01 diagonal matrix by broadcasting a 2D 01 diagonal matrix
    input_help = np.zeros((shape_matrix[-2], shape_matrix[-1]))
    for i in range(min(shape_matrix[-2], shape_matrix[-1])):
        input_help[i, i] = 1.0
    input_help = np.broadcast_to(input_help, shape_matrix)
    input_help = input_help.astype(dtype)
    if dtype == 'uint8':
        new_help = np.abs(input_help.astype('float16') - 1).astype(dtype)
    else:
        new_help = np.abs(input_help - 1)
        
    exp_output = input_matrix * new_help + input_help * new_input_diagonal
    # inputs and output to hold the data
    output = np.full(shape_matrix, np.nan, dtype)
    args = [input_matrix, input_diagonal, input_help, output]
    return args, exp_output, input_matrix, input_diagonal, input_help
