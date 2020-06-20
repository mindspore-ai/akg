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

"""run function for asin"""

import numpy as np

from akg.utils import kernel_exec as utils
from base import get_rtol_atol
from tensorio import compare_tensor
from test_op import asin


def asin_run(shape, dtype, attrs=None):
    mod = utils.op_build_test(asin.asin, [shape], [dtype],
                              kernel_name="asin", attrs=attrs)
    expect, inputs, output = gen_data(dtype, shape)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    rtol, atol = get_rtol_atol("asin", dtype)
    TestCase_Result = compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=False)

    return inputs, output, expect, TestCase_Result


def gen_data(dtype, shape):
    inputs = (np.random.rand(*shape) * 2 - 1).astype(dtype)  # domain of asin is [-1, 1]
    expect = np.arcsin(inputs)
    output = np.full(shape, np.nan, dtype)
    return expect, inputs, output
