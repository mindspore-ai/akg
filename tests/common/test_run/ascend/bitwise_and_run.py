# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""run function for bitwise_and"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import bitwise_and
from tests.common.base import get_rtol_atol
from akg.utils.dsl_create import produce_shapes


def bitwise_and_run(shape1, dtype1, shape2, dtype2, kernel_name, attrs):
    mod = utils.op_build_test(bitwise_and.bitwise_and,
                              [shape1, shape2], [dtype1, dtype2],
                              kernel_name=kernel_name, attrs=attrs)
    expect, inputs, output = gen_data(shape1, shape2, dtype1, dtype2)
    actual = utils.mod_launch(mod, (*inputs, output), expect=expect)

    rtol, atol = get_rtol_atol("bitwise_and", dtype1)
    testcase_result = compare_tensor(
        actual, expect, rtol=rtol, atol=atol, equal_nan=True)
    return input, actual, expect, testcase_result


def gen_data(shape1, shape2, dtype1, dtype2):
    int16_min = -32768
    int16_max = 32767
    uint16_min = 0
    uint16_max = 65535
    if dtype1 == "int16":
        x1 = np.random.randint(int16_min, int16_max, size=shape1).astype(dtype1)
        x2 = np.random.randint(int16_min, int16_max, size=shape2).astype(dtype2)
    elif dtype1 == "uint16":
        x1 = np.random.randint(uint16_min, uint16_max, size=shape1).astype(dtype1)
        x2 = np.random.randint(uint16_min, uint16_max, size=shape2).astype(dtype2)

    x1_min = int16_min if dtype1 == "int16" else uint16_min
    x1_max = int16_max if dtype1 == "int16" else uint16_max
    x2_min = int16_min if dtype2 == "int16" else uint16_min
    x2_max = int16_max if dtype2 == "int16" else uint16_max

    x1 = np.random.randint(x1_min, x1_max, size=shape1).astype(dtype1)
    x2 = np.random.randint(x2_min, x2_max, size=shape2).astype(dtype2)

    expect = np.bitwise_and(x1, x2)
    output = np.full(expect.shape, np.nan, dtype1)
    return expect, (x1, x2), output
