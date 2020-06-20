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
from test_op import batch_cholesky
def batch_cholesky_run(shape, dtype, attrs):
    # if 'tuning' in attrs.keys():
        # t = attrs.get("tuning", False)
        # kernel_name = attrs.get("kernel_name", False)
        # mod = utils.op_build(cholesky.cholesky, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        # if t:
            # exp_output, inputs, output = gen_data(dtype, shape)
            # return mod, exp_output, (inputs, output)
        # else:
            # return mod
    # else:
    # op_attrs=[shape, dtype]
    mod = utils.op_build(batch_cholesky.batch_cholesky, [shape], [dtype], kernel_name='cholesky', attrs=attrs)
    exp_output, inputs, output = gen_data(dtype, shape)
    #result_tvm
    acu_output=utils.mod_launch(mod,(inputs, output), expect=exp_output)
    #4) compare result
    print("-------hhhhhhhhhhhhhhhhhhhhhhhhh")
    # print(acu_output[255,:])
    # for i in range(shape[0]):
    #     for j in range(i):
    #         acu_output[i,j] = 0
    # dim = shape[0]
    # compare = inputs.copy()
    # for i in range(3):
    #     tmp = np.sqrt(compare[i,i])
    #     for j in range(dim):
    #         compare[i,j] = compare[i,j] / tmp
    #     tmp_vector = compare[i, : ].reshape([1,dim])
    #     tmp_matrix = np.transpose(tmp_vector)[i+1:, :].dot(tmp_vector)
    #     # if i != 1:
    #     compare[i+1:,:] = compare[i+1:, :] - tmp_matrix



    np.set_printoptions(suppress=True, precision=5)
    batch_size = shape[0]
    dim = shape[1]
    for i in range(batch_size):
        for j in range(dim):
            for k in range(j):
                acu_output[i,j,k] = 0
    # print(acu_output[255,:])
    # print('---- compare ----')
    # print(compare)
    # # print(compare[128:, 128:])
    # # print('---- acu_output ---')
    # print(acu_output)
    # print(acu_output[128:, 128:])

    # for i in range(255):
    #     compare[i+1:,i] = 0
    #     acu_output[i+1:,i] = 0
    # for i in range(dim):
    #     up = min(i, 2)
    #     for j in range(i):
    #         compare[i,j] = 0
    #         acu_output[i,j] = 0
    # for i in range(shape[0]):
    #     for j in range(i):
    #         acu_output[i,j] = 0
    # zero_count = 0
    # for i in range(dim):
    #     for j in range(1,dim):
    #         if np.abs(compare[i,j] - acu_output[i,j]) > 1e-5:
    #             print("error ", i,j,compare[i,j], acu_output[i,j])
    #             zero_count += 1
    # print(zero_count)
    TestCase_Result=np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)
    # TestCase_Result=np.allclose(acu_output, compare, rtol=5e-03, equal_nan=True)
    return inputs,acu_output,exp_output,TestCase_Result


def gen_data(dtype, shape):
    # 1)input data
    # inputs =  random_gaussian(shape, miu=1, sigma=10.0).astype(dtype)
    dim = shape[1]
    batch_size = shape[0]
    exp_output = np.zeros(shape).astype(dtype)
    inputs = np.zeros(shape).astype(dtype)

    for batch_idx in range(batch_size):
        a = np.random.rand(1, dim)
        b = np.transpose(a)
        batch_input = b.dot(a)
        batch_input = 0.01 * np.identity(dim) + batch_input
        batch_input = batch_input.astype(dtype)
        inputs[batch_idx, :, :] = batch_input
        batch_exp_out = np.linalg.cholesky(batch_input)
        batch_exp_out = np.transpose(batch_exp_out)
        batch_exp_out = batch_exp_out.astype(dtype)
        exp_output[batch_idx, :, :] = batch_exp_out
    # 2) except :Result_Numpy
    # exp_output = np.clip(inputs, min_val, max_val)
    # exp_output = np.linalg.cholesky(inputs)
    # exp_output = np.transpose(exp_output)

    # 3)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    print("************hhhhhhhhhhhhhhhhhhhhhhhhh*********", shape)
    return exp_output, inputs, output
