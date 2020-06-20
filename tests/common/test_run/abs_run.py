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
from akg.ops.math import abs
from base import get_rtol_atol

def abs_run(shape, dtype, attrs):

    # Result_Numpy
    input_shape = [shape]
    input_dtype = [dtype]

    op_attrs = [shape, dtype]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(abs.abs_value, input_shape, input_dtype, kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            exp_output, inputs, output = gen_date(dtype, shape)
            return mod, exp_output, (inputs, output)
        else:
            return mod
    else:
        # result_tvm
        mod = utils.op_build_test(abs.abs_value, input_shape, input_dtype, kernel_name='abs', attrs=attrs)
        exp_output, inputs, output = gen_date(dtype, shape)
        acu_output = utils.mod_launch(mod, (inputs, output), expect=exp_output)

        # compare result
        rtol, atol = get_rtol_atol("abs", dtype)
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=rtol, atol=atol, equal_nan=True)

        return inputs, acu_output, exp_output, TestCase_Result


def gen_date(dtype, shape):
    inputs = np.random.uniform(-1, 0, size=shape).astype(dtype)
    exp_output = np.abs(inputs)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    return exp_output, inputs, output
