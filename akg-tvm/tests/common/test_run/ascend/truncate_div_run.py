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

"""run test function for truncate_div"""

import numpy as np

from akg.utils import kernel_exec as utils
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.test_op.ascend import truncate_div
from tests.common.gen_random import random_gaussian


def truncate_div_run(shape1, dtype1, shape2, dtype2, attrs):
    """run function for truncate_div"""
    expect, inputs, output = gen_data(dtype1, dtype2, shape1, shape2)
    mod = utils.op_build_test(truncate_div.truncate_div,
                              [shape1, shape2], [dtype1, dtype2],
                              kernel_name="truncate_div",
                              attrs=attrs)
    output = utils.mod_launch(mod, (*inputs, output), expect=expect)
    rtol, atol = get_rtol_atol("truncate_div", dtype1)
    return inputs, output, expect, compare_tensor(
        output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype1, dtype2, shape1, shape2):
    """generate valid test data for truncate_div"""
    input1 = random_gaussian(shape1).astype(dtype1)
    input2 = random_gaussian(shape2).astype(dtype2)
    input2 = np.where(input2 == 0,
                      np.ones_like(input2),
                      input2)
    expect = np.divide(input1.astype("float32"), input2.astype("float32"))
    if dtype1 in ("int8", "uint8", "int32"):
        expect = np.trunc(expect).astype(dtype1)
    output = np.full(expect.shape, np.nan, dtype1)
    return expect, (input1, input2), output
