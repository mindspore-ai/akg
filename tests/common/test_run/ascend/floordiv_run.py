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
from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import FloorDiv
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol
from akg.utils.kernel_exec import product_is_mini

def floordiv_run(shape1, shape2, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(FloorDiv, [shape1, shape2], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, input1, input2, output = gen_data(dtype, shape1, shape2)
            return mod, expect, (input1, input2, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(FloorDiv, [shape1, shape2], [dtype, dtype], kernel_name=kernel_name, attrs=attrs)
        expect, input, input1, input2, output = gen_data(dtype, shape1, shape2)
        output = utils.mod_launch(mod, [input1, input2, output], expect=expect)
        rtol, atol = get_rtol_atol("floordiv", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape1, shape2):
    support_list = {"float16": np.float16, "float32": np.float32}
    input1 = random_gaussian(shape1, miu=1, sigma=0.1).astype(support_list[dtype])
    input2 = random_gaussian(shape2, miu=1, sigma=0.1).astype(support_list[dtype])
    np.where(input2 == 0, 0.1, input2)
    if product_is_mini():
        expect = 1.0 / input2 * input1
    else:
        expect = input1 / input2
    expect = np.floor(expect).astype(np.int32)
    input = list(input1)
    input.append(input2)
    output = np.full(expect.shape, np.nan, expect.dtype)
    return expect, input, input1, input2, output
