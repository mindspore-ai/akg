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

"""run function for xdivy"""

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import xdivy
from gen_random import random_gaussian
from base import get_rtol_atol
from akg.utils.dsl_create import produce_shapes


def xdivy_run(shape1, shape2, dtype, attrs):
    mod = utils.op_build_test(xdivy.xdivy, [shape1, shape2],
                              [dtype, dtype],
                              kernel_name="xdivy", attrs=attrs)
    expect, inputs, output = gen_data(shape1, shape2, dtype)
    output = utils.mod_launch(mod, (*inputs, output), expect=expect)
    rtol, atol = get_rtol_atol("xdivy", dtype)
    TestCase_Result = compare_tensor(
        output, expect, rtol=rtol, atol=atol, equal_nan=False)

    return inputs, output, expect, TestCase_Result


def gen_data(shape1, shape2, dtype):
    x1 = random_gaussian(shape1, miu=1, sigma=0.3).astype(dtype)
    x2 = random_gaussian(shape2, miu=1, sigma=0.3).astype(dtype)

    _, _, out_shape = produce_shapes(shape1, shape2)
    expect = np.where(np.equal(x1, 0.),
                      np.zeros_like(np.multiply(x1, x2)),
                      np.divide(x1, x2))
    output = np.full(out_shape, np.nan, dtype)
    return expect, (x1, x2), output
