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

"""run function for softplus"""

import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import softplus
from tests.common.gen_random import random_gaussian
from tests.common.base import get_rtol_atol


def softplus_run(shape, dtype, attrs):
    mod = utils.op_build_test(softplus.softplus, [shape], [dtype],
                              kernel_name="softplus", attrs=attrs)
    expect, inputs, output = gen_data(dtype, shape)
    output = utils.mod_launch(mod, (inputs, output), expect=expect)
    rtol, atol = get_rtol_atol("softplus", dtype)
    TestCase_Result = compare_tensor(
        output, expect, rtol=rtol, atol=atol, equal_nan=False)

    return inputs, output, expect, TestCase_Result


def gen_data(dtype, shape):
    inputs = random_gaussian(shape, miu=1, sigma=0.3).astype(dtype)
    expect = np.log1p(np.exp(-np.abs(inputs))) + np.maximum(inputs, 0)
    output = np.full(shape, np.nan, dtype)
    return expect, inputs, output
