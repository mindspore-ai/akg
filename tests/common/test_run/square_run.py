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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import square
from base import get_rtol_atol


def square_execute(shape, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = square_compile(shape, dtype, kernel_name, attrs, tuning=t)
        if t:
            args, exp_output, input = method_name(dtype, shape)
            return mod, exp_output, args
        else:
            return mod
    else:
        mod = square_compile(shape, dtype, kernel_name, attrs)
        args, exp_output, input = method_name(dtype, shape)
        acu_output = utils.mod_launch(mod, args, expect=exp_output)
        rtol, atol = get_rtol_atol("square", dtype)
        testcase_result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)
        return input, acu_output, exp_output, testcase_result


def method_name(dtype, shape):
    input = np.random.normal(size=shape).astype(dtype)
    exp_output = np.square(input)
    output = np.full(exp_output.shape, np.nan, dtype)
    args = [input, output]
    return args, exp_output, input


def square_compile(shape, dtype, kernel_name, attrs, tuning=False):
    return utils.op_build_test(square.square, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
