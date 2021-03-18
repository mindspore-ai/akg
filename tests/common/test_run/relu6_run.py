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
from tests.common.test_op import relu6
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian

def relu6_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(relu6.relu6, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            exp_output, input, output = gen_data(dtype, shape)
            return mod, exp_output, (input, output)
        else:
            return mod
    else:
        exp_output, input, output = gen_data(dtype, shape)
        mod = utils.op_build_test(relu6.relu6, [shape], [dtype], kernel_name='relu6', attrs=attrs)
        acu_output = utils.mod_launch(mod, (input, output), expect=exp_output)
        # compare result
        TestCase_Result = compare_tensor(acu_output, exp_output, rtol=5e-03, equal_nan=True)

        return input, acu_output, exp_output, TestCase_Result


def gen_data(dtype, shape):
    # Result_Numpy
    input = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    zero = np.full(shape, 0, dtype)
    six = np.full(shape, 6, dtype)
    max = np.maximum(input, zero)
    exp_output = np.minimum(max, six)
    # inputs and output to hold the data
    output = np.full(shape, np.nan, dtype)
    return exp_output, input, output
