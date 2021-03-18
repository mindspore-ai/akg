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
from tests.common.test_op import reduction_layer
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def reduction_layer_execute(shape, dtype, axis, op, coeff, attrs):
    exp_output, inputs, args = gen_data(shape, dtype, axis, op, coeff)
    mod = reduction_layer_compile(shape, dtype, axis, op, coeff, attrs)
    # result_tvm
    acu_output = utils.mod_launch(mod, args, expect=exp_output)

    # compare result
    rtol, atol = get_rtol_atol("reduction_layer", dtype)
    TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return inputs, acu_output, exp_output, TestCase_Result

def gen_data(shape, dtype, axis, op, coeff):
    # Result_Numpy
    inputs = random_gaussian(shape, miu=0, sigma=0.3).astype(dtype)
    
    axis = list(range(axis, len(shape)))
    for i, _ in enumerate(axis):
        if axis[i] < 0:
            axis[i] = axis[i] + len(shape)
    axis = set(axis)
    axis = list(axis)
    axis.sort()
    
    if op == "SUM":
        tmp = np.multiply(inputs, coeff)
    elif op == "ASUM":
        tmp = np.multiply(np.abs(inputs), coeff)
    elif op == "SUMSQ":
        tmp = np.multiply(np.multiply(inputs, inputs), coeff)
    elif op == "MEAN":
        size = 1
        for i, _ in enumerate(axis):
            size = size * shape[axis[i]]
        tmp = np.multiply(inputs, float(coeff) / size)
    
    res = np.sum(tmp, tuple(axis))
    exp_output = res

    # inputs and output to hold the data
    output = np.full((1,) if res.shape == () else res.shape, np.nan, dtype)
    args = []
    args.append(inputs)
    args.append(output)
    return exp_output, inputs, args

def reduction_layer_compile(shape, dtype, axis, op, coeff, attrs, 
                                  kernel_name='reduction_layer', runing=False):
    return utils.op_build_test(reduction_layer.reduction_layer, [shape], [dtype], 
                               [axis, op, coeff], kernel_name=kernel_name, attrs=attrs, tuning=runing)
