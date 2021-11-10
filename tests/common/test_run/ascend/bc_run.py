# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import bc_test
from tests.common.gen_random import random_gaussian


def bc_run(shape1, shape2, shape3, dtype, kernel_name="bc_test", attrs={}, polyhedral=True):
    input_shape = [shape1, shape2, shape3]
    input_dtype = [dtype, dtype, dtype]
    op_attrs = []

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(bc_test.bc_test, input_shape, input_dtype, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, polyhedral=polyhedral, tuning=t)
        if t:
            expect, input1, input2, input3, output = gen_data(dtype, shape1, shape2, shape3)
            return mod, expect, (input1, input2, input3, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(bc_test.bc_test, input_shape, input_dtype, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, polyhedral=polyhedral)
        expect, input1, input2, input3, output = gen_data(dtype, shape1, shape2, shape3)
        output = utils.mod_launch(mod, (input1, input2, input3, output), expect=expect)
        return (input1, input2, input3), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, shape1, shape2, shape3):
    input1 = random_gaussian(shape1, miu=1, sigma=0.1)
    input2 = random_gaussian(shape2, miu=1, sigma=0.1)
    input3 = random_gaussian(shape3, miu=1, sigma=0.1)
    if (dtype == "int32"):
        input1 = input1.astype(np.int32)
        input2 = input2.astype(np.int32)
        input3 = input3.astype(np.int32)
    elif (dtype == "float16"):
        input1 = input1.astype(np.float16)
        input2 = input2.astype(np.float16)
        input3 = input3.astype(np.float16)
    expect = np.add(input1, input3) + np.subtract(input1, input3) + np.add(input2, input3) + np.add(input1, input2)
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    return expect, input1, input2, input3, output
