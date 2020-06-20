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

"""run function for softplus_grad"""

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import softplus_grad
from gen_random import random_gaussian
from .conv_utils import random_gaussian as random_gaussian_pn
from base import get_rtol_atol


def softplus_grad_run(shape, dtype, attrs):
    mod = utils.op_build_test(softplus_grad.softplus_grad,
                              [shape, shape], [dtype, dtype],
                              kernel_name="softplus_grad", attrs=attrs)

    expect, inputs, output = gen_data(shape, dtype)
    output = utils.mod_launch(mod, (*inputs, output), expect=expect)
    rtol, atol = get_rtol_atol("softplus_grad", dtype)
    TestCase_Result = compare_tensor(
        output, expect, rtol=rtol, atol=atol, equal_nan=False)
    return inputs, output, expect, TestCase_Result


def gen_data(shape, dtype):
    dy = random_gaussian_pn(shape).astype(dtype)
    if dtype == "uint8":
        x = np.abs(random_gaussian(shape, miu=1, sigma=0.3)).astype(dtype)
    else:
        x = random_gaussian_pn(shape).astype(dtype)
    dx = np.divide(dy.astype(np.float32) * np.exp(x.astype(np.float32)),
                   1 + np.exp(x.astype(np.float32))).astype(dtype)
    output = np.full(shape, np.nan, dtype)
    return dx, (dy, x), output
