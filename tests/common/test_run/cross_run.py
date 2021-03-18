# Copyright 2020 Huawei Technologies Co., Ltd
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

"""run test function for cross"""

import numpy as np

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.test_op import cross
from tests.common.gen_random import random_gaussian


def cross_run(shape1, dtype1, shape2, dtype2, attrs):
    """run function for cross"""
    mod = utils.op_build_test(cross.cross,
                              [shape1, shape2], [dtype1, dtype2],
                              kernel_name="cross",
                              attrs=attrs)
    expect, inputs, out_buf = gen_data(dtype1, dtype2, shape1, shape2)
    output = utils.mod_launch(mod, (*inputs, out_buf), expect=expect)
    rtol, atol = get_rtol_atol("cross", dtype1)
    cmp_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    return inputs, output, expect, cmp_res

def gen_data(dtype1, dtype2, shape1, shape2):
    """generate valid test data for cross"""
    input1 = random_gaussian(shape1).astype(dtype1)
    input2 = random_gaussian(shape2).astype(dtype2)

    if dtype1 in ("int8", "uint8", "int32"):
        # for overflow case, numpy will truncate the result, but davinci will
        # make it maximum or minimuam value.
        expect = np.cross(input1.astype("float32"), input2.astype("float32"),
                          axisa=0, axisb=0, axisc=0)
        expect = np.maximum(expect, np.ones_like(expect) * np.iinfo(dtype1).min)
        expect = np.minimum(expect, np.ones_like(expect) * np.iinfo(dtype1).max)
        expect = expect.astype(dtype1)
    else:
        expect = np.cross(input1, input2, axisa=0, axisb=0, axisc=0)

    out_buf = np.full(expect.shape, np.nan, dtype1)
    return expect, (input1, input2), out_buf
