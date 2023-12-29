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
from akg.ops.math import mul
from akg.ops.math import sub
from tests.common.gen_random import random_gaussian


def mul_sub(first_input, second_input, third_input, target="cce"):
    temp = mul(first_input, second_input, target=target)
    output = sub(temp, third_input, target=target)
    return output


def fused_mul_sub_execute(shape1, shape2, shape3, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = fused_mul_sub_compile(shape1, shape2, shape3, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input1, input2, input3, output = gen_data(shape1, shape2, shape3, dtype)
            return mod, expect, (input1, input2, input3, output)
        else:
            return mod
    else:
        expect, input1, input2, input3, output = gen_data(shape1, shape2, shape3, dtype)
        mod = fused_mul_sub_compile(shape1, shape2, shape3, dtype, attrs)
        output = utils.mod_launch(mod, (input1, input2, input3, output), expect=expect)
        return (input1, input2, input3), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def fused_mul_sub_compile(shape1, shape2, shape3, dtype, attrs, kernel_name="fused_mul_sub", tuning=False):
    op_attrs = []
    return utils.op_build_test(mul_sub, [shape1, shape2, shape3], [dtype, dtype, dtype], op_attrs,
                               kernel_name=kernel_name, attrs=attrs, tuning=tuning)


def gen_data(shape1, shape2, shape3, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    input3 = random_gaussian(shape3, miu=1, sigma=0.1).astype(support_list[dtype])
    temp = np.multiply(input1, input2)
    expect = np.subtract(temp, input3)
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    return expect, input1, input2, input3, output
