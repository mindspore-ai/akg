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
from tests.common.test_op.ascend import cholesky


def cholesky_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build(cholesky.cholesky, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            exp_output, inputs, output = gen_data(dtype, shape)
            return mod, exp_output, (inputs, output)
        else:
            return mod
    else:
        # op_attrs=[shape, dtype]
        mod = utils.op_build(cholesky.cholesky, [shape], [dtype], kernel_name='cholesky', attrs=attrs)
        exp_output, inputs, output = gen_data(dtype, shape)
        # result_tvm
        acu_output = utils.mod_launch(mod, (inputs, output), expect=exp_output)
        # 4) compare result
        TestCase_Result = np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)

        return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shape):
    # 1)input data
    # inputs =  random_gaussian(shape, miu=1, sigma=10.0).astype(dtype)
    dim = shape[0]
    a = np.random.rand(1, dim)
    b = np.transpose(a)
    inputs = b.dot(a)
    inputs = 0.01 * np.identity(dim) + inputs
    # 2) except :Result_Numpy
    # exp_output = np.clip(inputs, min_val, max_val)
    exp_output = np.linalg.cholesky(inputs)
    # 3)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    print("************hhhhhhhhhhhhhhhhhhhhhhhhh*********", shape)
    return exp_output, inputs, output
