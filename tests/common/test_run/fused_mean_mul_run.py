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
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.math import mul, mean
from tests.common.gen_random import random_gaussian

def mean_mul(first_input, second_input, axis=None, keepdims=False):
    temp, _ = mean.mean(first_input, axis, keepdims)
    output = mul.mul(temp, second_input)
    return output


def fused_mean_mul_execute(shape1, shape2, dtype, axis, keepdims, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = fused_mean_mul_compile(shape1, shape2, dtype, axis, keepdims, kernel_name, attrs, tuning=t)
        if t:
            expect, input1, input2, output = gen_data(shape1, shape2, dtype, axis, keepdims)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        expect, input1, input2, output = gen_data(shape1, shape2, dtype, axis, keepdims)
        mod = fused_mean_mul_compile(shape1, shape2, dtype, axis, keepdims, kernel_name, attrs)
        output = utils.mod_launch(mod, (input1, input2, output), expect=expect)
        return (input1, input2), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def fused_mean_mul_compile(shape1, shape2, dtype, axis, keepdims, kernel_name, attrs, tuning=False):
    op_attrs = [axis, keepdims]
    return utils.op_build_test(mean_mul, [shape1, shape2], [dtype, dtype], op_attrs, kernel_name=kernel_name, attrs=attrs, tuning=tuning)


def gen_data(shape1, shape2, dtype, axis=None, keepdims=False):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    temp = np.mean(input1, axis=axis, keepdims=keepdims)
    expect = np.multiply(temp, input2)
    out_shape = expect.shape
    output = np.full(out_shape, np.nan, dtype)
    return expect, input1, input2, output
