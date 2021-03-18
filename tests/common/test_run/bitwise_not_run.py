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
from tests.common.test_op import bitwise_not


def bitwise_not_run(shape, dtype, kernel_name, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(bitwise_not.bitwise_not, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, expect, input = gen_data(dtype, shape)
            return mod, expect, args
        else:
            return mod
    else:
        mod = utils.op_build_test(bitwise_not.bitwise_not, [shape], [dtype], kernel_name=kernel_name, attrs=attrs)
        args, expect, input = gen_data(dtype, shape)
        actual = utils.mod_launch(mod, args, expect=expect)
        testcase_result = compare_tensor(actual, expect, rtol=5e-03, equal_nan=True)
        return input, actual, expect, testcase_result


def gen_data(dtype, shape):
    if dtype == "int32":
        input = np.random.randint(-512, 512, size=shape).astype(dtype)
    elif dtype == "int8":
        input = np.random.randint(-128, 127, size=shape).astype(dtype)
    add_one = np.add(input, 1)
    expect = np.multiply(add_one, -1)
    output = np.full(expect.shape, np.nan, dtype)
    args = [input, output]
    return args, expect, input
