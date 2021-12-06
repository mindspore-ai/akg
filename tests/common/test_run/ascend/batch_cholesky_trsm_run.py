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

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import batch_cholesky_trsm

def batch_cholesky_trsm_run(shape1, shape2, dtype, attrs):
    mod = utils.op_build(batch_cholesky_trsm.batch_cholesky_trsm, [shape1, shape2], [dtype,dtype], kernel_name='batch_cholesky_trsm', attrs=attrs)
    exp_output, inputs1, inputs2, output = gen_data(dtype, shape1, shape2)
    #result_tvm
    acu_output=utils.mod_launch(mod,(inputs1, inputs2, output))
    # np.set_printoptions(suppress=True, precision=5)
    # batch_size = shape1[0]
    # dim = shape1[1]
    # for i in range(batch_size):
    #     for j in range(dim):
    #         for k in range(j):
    #             acu_output[i,j,k] = 0
    # dim = shape1[1]
    # acu_output[0,:,:] = np.linalg.solve(acu_output[0,:,:], np.identity(dim))
    #acu_output = acu_output[0]
    print("====")
    print(inputs1[0,:,:])
    print("====")
    print(acu_output[0,:,:])
    print("====")
    print(exp_output[0,:,:])

    TestCase_Result=np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)
    return inputs1,acu_output,exp_output,TestCase_Result


def gen_data(dtype, shape1, shape2):
    dim = shape1[1]
    batch_size = shape1[0]
    exp_output = np.zeros(shape1).astype(dtype)
    inputs1 = np.zeros(shape1).astype(dtype)
    inputs2 = np.zeros(shape2).astype(dtype)
    for batch_idx in range(batch_size):
        a = np.random.rand(1, dim)
        b = np.transpose(a)
        batch_input = b.dot(a)
        batch_input = 0.01 * np.identity(dim) + batch_input
        inputs1[batch_idx, :, :] = batch_input.copy()
        inputs2[batch_idx, :, :] = np.identity(dim).astype(dtype)

        batch_exp_out = np.linalg.cholesky(batch_input)
        batch_exp_out = np.transpose(batch_exp_out)
        batch_exp_out = np.linalg.solve(batch_exp_out, inputs2[batch_idx, :, :])
        batch_exp_out = batch_exp_out.astype(dtype)
        exp_output[batch_idx, :, :] = batch_exp_out

    output = np.full(shape1, np.nan, dtype)
    print("************hhhhhhhhhhhhhhhhhhhhhhhhh*********", shape1)
    return exp_output, inputs1, inputs2, output

