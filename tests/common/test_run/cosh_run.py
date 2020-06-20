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

import numpy as np
from akg.utils import kernel_exec as utils
from test_op import cosh
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def cosh_execute(shape, dtype, attrs):
    exp_output, inputs, args = gen_data(dtype, shape)
    mod = cosh_compile(shape, dtype, attrs)
    # result_tvm
    acu_output = utils.mod_launch(mod, args, expect=exp_output)

    # compare result
    rtol, atol = get_rtol_atol("cosh", dtype)
    TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shape):
    # Result_Numpy
    inputs = random_gaussian(shape, miu=0, sigma=0.3).astype(dtype)
    exp_output = np.cosh(inputs)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = []
    args.append(inputs)
    args.append(output)
    return exp_output, inputs, args


def cosh_compile(shape, dtype, attrs, kernel_name='cosh', runing=False):
    return utils.op_build_test(cosh.cosh, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=runing)
