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
from test_op import SecondOrder_trace_extract
from gen_random import random_gaussian

def trace_extract_run(shape, dtype, attrs):
    """
    ops run func.
    """
    mod = utils.op_build(SecondOrder_trace_extract.trace_extract, [shape], [dtype], kernel_name='trace', attrs=attrs)
    exp_output, inputs, output = gen_data(dtype, shape)
    #result_tvm
    acu_output=utils.mod_launch(mod,(inputs, output), expect=exp_output)
    # 4) compare result
    print('----result----')
    print(acu_output)
    print('----compare---')
    print(exp_output)
    TestCase_Result=np.allclose(acu_output, exp_output, rtol=5e-03, equal_nan=True)

    return inputs,acu_output,exp_output,TestCase_Result


def gen_data(dtype, shape1):
    """
    generate data.
    """
    inputs =  random_gaussian(shape1, miu=1, sigma=10.0).astype(dtype)
    dim = inputs.shape[1]
    exp_output = []
    for i in range(dim):
        exp_output.append(inputs[0,i,i])
    exp_output = np.asarray(exp_output).reshape([1,dim])
    output = np.full((1,dim), np.nan, dtype)
    return exp_output, inputs, output
