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
acos_grad run define
"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import acos_grad
from tests.common.gen_random import random_gaussian

def acos_grad_run(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(acos_grad.acos_grad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, grad, inputs, output = gen_data(dtype, shape)
            return mod, expect, (inputs, grad, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(acos_grad.acos_grad, [shape, shape], [dtype, dtype], kernel_name='acos_grad', attrs=attrs)
        expect, grad, inputs, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (inputs, grad, output), expect=expect)
        # compare result
        TestCase_Result = compare_tensor(output, expect, rtol=5e-03, atol=1e-04, equal_nan=False)

        return (inputs, grad), output, expect, TestCase_Result


def gen_data(dtype, shape):
    # Generate data for testing the op
    inputs = random_gaussian(shape, miu=0, sigma=0.1).astype(dtype)
    grad = random_gaussian(shape, miu=0, sigma=0.1).astype(dtype)
    expect = - (1 / np.sqrt(1 - np.square(inputs))) * grad
    # inputs and output to hold the data
    output = np.full(expect.shape, np.nan, dtype)
    return expect, grad, inputs, output
