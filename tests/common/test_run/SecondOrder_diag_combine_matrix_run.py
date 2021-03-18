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

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op import SecondOrder_diag_combine_matrix
from tests.common.gen_random import random_gaussian

def diag_combine_matrix_run(shape, dtype, attrs):
    """
    ops run func.
    """
    if len(shape) == 1:
        shape = shape[0]
        mod = utils.op_build_test(SecondOrder_diag_combine_matrix.diag_combine_matrix_1, [shape], [dtype], kernel_name='diag_combine_matrix', attrs=attrs)
        exp_output, inputs, output = gen_data1(dtype, shape)
        acu_output = utils.mod_launch(mod, (inputs, output), expect=exp_output)
        TestCase_Result=np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)
        return inputs,acu_output,exp_output,TestCase_Result
    else:
        print(len(shape))
        input_shape = []
        input_dtype = []
        for val in shape:
            input_shape.append(val)
            input_dtype.append(dtype)
        exp_output, inputs, output = gen_data2(input_dtype, input_shape)
        input1, input2 = inputs
        mod = utils.op_build_test(SecondOrder_diag_combine_matrix.diag_combine_matrix_2, input_shape, input_dtype, kernel_name='diag_combine_matrix', attrs=attrs)
        acu_output = utils.mod_launch(mod, (input1, input2, output), expect=exp_output)
        TestCase_Result=np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)
        return input1,acu_output,exp_output,TestCase_Result

def gen_data1(dtype, shape):
    """
    generate data.
    """
    inputs =  random_gaussian(shape, miu=1, sigma=10.0).astype(dtype)
    batch_dim = shape[1]
    batch_size = shape[0]
    matrix_dim = batch_size * batch_dim
    exp_output = np.zeros((matrix_dim, matrix_dim)).astype(dtype)
    for i in range(batch_size):
        for j in range(batch_dim):
            for k in range(batch_dim):
                exp_output[i * batch_dim + j, i * batch_dim + k] = inputs[i, j, k]
    output = np.full((matrix_dim,matrix_dim), np.nan, dtype)
    return exp_output, inputs, output

def gen_data2(dtype, shape):
    """
    generate data.
    """
    input_matrix_num = len(dtype)
    inputs = []
    matrix_dim = 0

    for i in range(input_matrix_num):
        shape_ = shape[i]
        dtype_ = dtype[i]
        matrix_dim += shape_[1] * shape_[0]
        inputs_tmp = random_gaussian(shape_, miu=1, sigma=10.0).astype(dtype_)
        inputs.append(inputs_tmp)
    output = np.full((matrix_dim,matrix_dim), np.nan, dtype[0])
    dim = 0
    exp_output = np.zeros((matrix_dim, matrix_dim)).astype(dtype[0])
    for i in range(input_matrix_num):
        batch_size = shape[i][0]
        batch_dim = shape[i][1]
        for j in range(batch_size):
            for m in range(batch_dim):
                for n in range(batch_dim):
                    exp_output[dim + j * batch_dim + m, dim + j * batch_dim + n] = inputs[i][j,m,n]
        dim += batch_dim * batch_size
    return exp_output, inputs, output
