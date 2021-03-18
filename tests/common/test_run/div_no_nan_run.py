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
from tests.common.test_op import div_no_nan
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def div_no_nan_execute(shapes, dtype, attrs):
    exp_output, inputs, args = gen_data(dtype, shapes)
    mod = div_no_nan_compile(shapes, dtype, attrs)
    # result_tvm
    acu_output = utils.mod_launch(mod, args, expect=exp_output)
    # compare result
    rtol, atol = get_rtol_atol("div_no_nan", dtype)
    TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shapes):
    # Result_Numpy
    data_x = random_gaussian(shapes[0], miu=1, sigma=0.1).astype(dtype)
    data_y = random_gaussian(shapes[1], miu=0, sigma=2**-64).astype(dtype)
    if dtype in ["uint8", "int8", "int32"]:
        is_zero = np.equal(0, data_y)
    if dtype in ["float16"]:
        is_zero = np.less(np.abs(data_y), 2**-12)
    if dtype in ["float32"]:
        is_zero = np.less(np.abs(data_y), 2**-64) 
    if dtype in ["uint8", "int8", "int32"]:
        exp_output = np.floor_divide(np.multiply(data_x, (1 - is_zero)), data_y + is_zero)
    if dtype in ["float16", "float32"]:
        exp_output = np.true_divide(np.multiply(data_x, (1 - is_zero)), data_y + is_zero)
    # inputs and output to hold the data
    output = np.full(exp_output.shape, np.nan, dtype)
    inputs = [data_x, data_y]
    args = [data_x, data_y, output]
    return exp_output, inputs, args


def div_no_nan_compile(shapes, dtype, attrs, kernel_name='div_no_nan', runing=False):
    return utils.op_build_test(div_no_nan.div_no_nan, [shapes[0], shapes[1]], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=runing)
