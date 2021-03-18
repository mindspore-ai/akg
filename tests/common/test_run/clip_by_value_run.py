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
from tests.common.test_op import clip_by_value
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def clip_by_value_execute(shapes, dtype, attrs):
    exp_output, inputs, args = gen_data(dtype, shapes)
    mod = clip_by_value_compile(shapes, dtype, attrs)
    # result_tvm
    acu_output = utils.mod_launch(mod, args, expect=exp_output)

    # compare result
    rtol, atol = get_rtol_atol("clip_by_value", dtype)
    TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

    return inputs, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shapes):
    # Result_Numpy
    data = random_gaussian(shapes[0], miu=0, sigma=1).astype(dtype)
    clip_min_value = random_gaussian(shapes[1], miu=-1, sigma=0.1).astype(dtype)
    clip_max_value = clip_min_value + 2
    res_max = np.maximum(data, clip_min_value)
    exp_output = np.minimum(res_max, clip_max_value)
    # inputs and output to hold the data
    output = np.full(shapes[0], np.nan, dtype)
    inputs = [data, clip_min_value, clip_max_value]
    args = [data, clip_min_value, clip_max_value, output]
    return exp_output, inputs, args


def clip_by_value_compile(shapes, dtype, attrs, kernel_name='clip_by_value', runing=False):
    return utils.op_build_test(clip_by_value.clip_by_value, [shapes[0], shapes[1], shapes[2]], [dtype, dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=runing)
