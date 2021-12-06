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
from tests.common.test_op.ascend import SecondOrder_diag_split_matrix
from tests.common.gen_random import random_gaussian
split_dim = 128

def diag_split_matrix_run(shape, dtype, attrs):
    """
    ops run func.
    """
    dim = shape[0]
    if (dim // split_dim) > 32:
        mod = utils.op_build_test(SecondOrder_diag_split_matrix.diag_split_matrix_4608, [shape], [dtype], kernel_name='trace', attrs=attrs)
        exp_output, inputs, out1, out2 = gen_data1(dtype, shape)
        acu_output1, acu_output2 = utils.mod_launch(mod, (inputs, out1, out2), (-2, -1), expect=exp_output)
        print("=====",dim," compare====")
        print(acu_output1.shape)
        print(acu_output2.shape)
        print("=====",dim," compare====")
        acu_output = np.concatenate((acu_output1, acu_output2), axis = 0 )
        TestCase_Result=np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)
        return inputs,acu_output,exp_output,TestCase_Result
    elif dim == 576:
        mod = utils.op_build_test(SecondOrder_diag_split_matrix.diag_split_matrix_576, [shape], [dtype], kernel_name='trace', attrs=attrs)
        exp_output1, exp_output2, inputs, out1, out2 = gen_data3(dtype, shape)
        acu_output1, acu_output2 = utils.mod_launch(mod, (inputs, out1, out2), (-2, -1), expect=exp_output1)
        print("=====",dim," compare====")
        print(acu_output1.shape)
        print(acu_output2.shape)
        print("=====",dim," compare====")
        # acu_output = np.concatenate((acu_output1, acu_output2), axis = 0 )
        TestCase_Result=np.allclose(acu_output1, exp_output1, rtol=5e-03, equal_nan=True)
        TestCase_Result=np.allclose(acu_output2, exp_output2, rtol=5e-03, equal_nan=True)
        return inputs,acu_output1,exp_output1,TestCase_Result
    else:
        mod = utils.op_build(SecondOrder_diag_split_matrix.diag_split_matrix_small, [shape], [dtype], kernel_name='trace01', attrs=attrs)
        exp_output, inputs, out1 = gen_data2(dtype, shape)
        acu_output = utils.mod_launch(mod, (inputs, out1), expect=exp_output)
        print("=====",dim," compare====")
        print(acu_output.shape)
        print("=====",dim," compare====")
        TestCase_Result=np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)
        return inputs,acu_output,exp_output,TestCase_Result

def gen_data1(dtype, shape1):
    """
    generate data.
    """
    inputs =  random_gaussian(shape1, miu=1, sigma=10.0).astype(dtype)
    dim = inputs.shape[0]
    split_num = dim // split_dim
    exp_output = np.zeros((split_num, split_dim, split_dim)).astype(dtype)
    for i in range(split_num):
        for j in range(split_dim):
            for k in range(split_dim):
                exp_output[i,j,k] = inputs[i*split_dim + j, i*split_dim+k]

    out1 = np.full((32,split_dim, split_dim), np.nan, dtype)
    out2 = np.full((4, split_dim, split_dim), np.nan, dtype)
    return exp_output, inputs, out1, out2

def gen_data2(dtype, shape1):
    """
    generate data.
    """
    inputs =  random_gaussian(shape1, miu=1, sigma=10.0).astype(dtype)
    dim = inputs.shape[0]
    split_num = dim // split_dim
    exp_output = np.zeros((split_num, split_dim, split_dim)).astype(dtype)
    for i in range(split_num):
        for j in range(split_dim):
            for k in range(split_dim):
                exp_output[i,j,k] = inputs[i*split_dim + j, i*split_dim+k]

    out1 = np.full((split_num,split_dim, split_dim), np.nan, dtype)
    return exp_output, inputs, out1

def gen_data3(dtype, shape1):
    """
    generate data.
    """
    inputs =  random_gaussian(shape1, miu=1, sigma=10.0).astype(dtype)
    dim = inputs.shape[0]
    split_num = dim // split_dim
    exp_output1 = np.zeros((split_num, split_dim, split_dim)).astype(dtype)
    for i in range(split_num):
        for j in range(split_dim):
            for k in range(split_dim):
                exp_output1[i,j,k] = inputs[i*split_dim + j, i*split_dim+k]

    exp_output2 = np.zeros((1, dim - split_dim * split_num, dim - split_dim * split_num)).astype(dtype)
    for i in range(dim - split_dim * split_num):
        for j in range(dim - split_dim * split_num):
            exp_output2[0,i,j] = inputs[split_num * split_dim + i, split_num * split_dim + j]



    out1 = np.full((split_num,split_dim, split_dim), np.nan, dtype)
    out2 = np.full((1,dim - split_dim * split_num, dim - split_dim * split_num), np.nan, dtype)
    return exp_output1, exp_output2, inputs, out1, out2
