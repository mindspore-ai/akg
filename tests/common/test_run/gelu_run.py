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
from test_op import gelu
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def gelu_execute(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = gelu_compile(shape, dtype, attrs, kernel_name=kernel_name, runing=t)
        if t:
            exp_output, inputs, args = gen_data(dtype, shape)
            return mod, exp_output, args
        else:
            return mod
    else:
        exp_output, inputs, args = gen_data(dtype, shape)
        mod = gelu_compile(shape, dtype, attrs)
        # result_tvm
        acu_output = utils.mod_launch(mod, args, expect=exp_output)

        # compare result
        rtol, atol = get_rtol_atol("gelu", dtype)
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

        return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shape):
    # Result_Numpy
    inputs = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (inputs + 0.044715 * np.power(inputs, 3)))))
    # cdf = 0.5 * (1.0 + np.tanh((0.7978845 * (inputs + 0.044715 * np.power(inputs, 3)))))
    exp_output = inputs * cdf
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = []
    args.append(inputs)
    args.append(output)
    return exp_output, inputs, args


def gelu_compile(shape, dtype, attrs, kernel_name='gelu', runing=False):
    return utils.op_build_test(gelu.gelu, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=runing)
