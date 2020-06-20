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
from akg.ops.math import addn
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def addn_execute(shape, dtype, n, attrs={}):
    # for i in range(len(shapes)):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod, shapes = addn_compile(shape, dtype, n, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, inputs, args = gen_data(dtype, n, shape, shapes)
            return mod, expect, args
        else:
            return mod
    else:
        # Result_addn
        mod, shapes = addn_compile(shape, dtype, n, attrs)
        exp_output, inputs, args = gen_data(dtype, n, shape, shapes)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)
        # compare result
        rtol, atol = get_rtol_atol("addn", dtype)
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)
        return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, n, shape, shapes):
    inputs = []
    for i in range(n):
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
        inputs.append(input)
        shapes.append(shape)
    exp_output = np.sum(inputs, axis=0)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    args = inputs
    args.append(output)
    return exp_output, inputs, args


def addn_compile(shape, dtype, n, attrs, kernel_name="addn", tuning=False):
    shapes = []
    for i in range(n):
        shapes.append(shape)
    return utils.op_build_test(addn.addn, [shapes], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning), shapes
