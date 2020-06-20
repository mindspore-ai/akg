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
from test_op import less_equal
from tensorio import compare_tensor
from base import get_rtol_atol
from gen_random import random_gaussian

def less_equal_execute(shapes, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = less_equal_compile(shapes, dtype, kernel_name, attrs, tuning=t)
        if t:
            expect, inputs, output = gen_data(dtype, shapes)
            return mod, expect, (inputs + output)
        else:
            return mod
    else:
        mod = less_equal_compile(shapes, dtype, kernel_name, attrs)
        expect, inputs, output = gen_data(dtype, shapes)
        output = utils.mod_launch(mod, inputs + [output], expect=expect)
        rtol, atol = get_rtol_atol("less_equal", dtype)
        return inputs, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shapes):
    inputs = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input = random_gaussian(shape, miu=1, sigma=(0.1 + 100 * i)).astype(dtype)
        inputs.append(input)
    if len(inputs) != 2:
        raise RuntimeError("inputs num should be 2")
    expect = np.less_equal(inputs[0], inputs[1])
    output = np.full(expect.shape, np.nan, "bool")
    return expect, inputs, output


def less_equal_compile(shapes, dtype, kernel_name, attrs, tuning=False):
    return utils.op_build_test(less_equal.less_equal, shapes, [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
