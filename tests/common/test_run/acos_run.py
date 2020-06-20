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

"""
acos run define
"""

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import acos
from gen_random import random_gaussian

def acos_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(acos.acos, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(dtype, shape)
            return mod, expect, (inputs, output)
        else:
            return mod
    else:
        expect, inputs, output = gen_data(dtype, shape)
        # op build
        mod = utils.op_build_test(acos.acos, [shape], [dtype], kernel_name='acos', attrs=attrs)

        # inputs and output to hold the data and result_tvm
        output = utils.mod_launch(mod, (inputs, output), expect=expect)

        # compare result
        TestCase_Result = compare_tensor(output, expect, rtol=1e-03, atol=1e-04, equal_nan=False)

        return inputs, output, expect, TestCase_Result


def gen_data(dtype, shape):
    # Generate data for testing the op
    inputs = random_gaussian(shape, miu=0, sigma=0.1).astype(dtype)
    expect = np.arccos(inputs)
    output = np.full(shape, np.nan, dtype)
    return expect, inputs, output
