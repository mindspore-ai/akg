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
from tests.common.test_op.ascend import eltwise
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def eltwise_execute(shape, dtype, n, mod, coeff, attrs):
    # Result_eltwise
    module = eltwise_compile(shape, dtype, n, mod, coeff, attrs)
    exp_output, inputs, args = gen_data(shape, dtype, n, mod, coeff)
    acu_output = utils.mod_launch(module, tuple(args), expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("eltwise", dtype)
    TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)
    return inputs, acu_output, exp_output, TestCase_Result


def gen_data(shape, dtype, n, mod, coeff):
    inputs = []
    for i in range(n):
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
        inputs.append(input)
    if mod == 0:
        exp_output = inputs[0]
        for i in range(1, n):
            exp_output = np.multiply(exp_output, inputs[i])
    if mod == 1 and len(coeff) == 0:
        exp_output = np.sum(inputs, axis=0)
    if mod == 1 and len(coeff) == n:
        exp_output = inputs[0] * coeff[0]
        for i in range(1, n):
            exp_output = exp_output + inputs[i] * coeff[i]
    if mod == 2:
        exp_output = inputs[0]
        for i in range(1, n):
            exp_output = np.maximum(exp_output, inputs[i]) 
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = inputs + [output]
    return exp_output, inputs, args

def eltwise_compile(shape, dtype, n, mod, coeff, attrs, kernel_name="eltwise", tuning=False):
    shapes = []
    for i in range(n):
        shapes.append(shape)   
    return utils.op_build_test(eltwise.eltwise, [shapes], [dtype], op_attrs=[mod, coeff], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
