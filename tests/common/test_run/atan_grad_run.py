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

"""run function for gradient of arctangent"""

import numpy as np

from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from tests.common.test_op import atan_grad

def atan_grad_run(shape1, dtype1, shape2, dtype2, attrs):
    """run function for arctangent"""
    mod = utils.op_build_test(atan_grad.atan_grad,
                              [shape1, shape2], [dtype1, dtype2],
                              kernel_name="atan_grad", attrs=attrs)
    expect, inputs, out_buf = gen_data(shape1, dtype1, shape2, dtype2)
    output = utils.mod_launch(mod, (*inputs, out_buf), expect=expect)
    rtol, atol = get_rtol_atol("atan_grad", dtype1)
    cmp_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    return inputs, output, expect, cmp_res

def gen_data(shape1, dtype1, shape2, dtype2):
    """generate valid data for arctangent"""
    head = random_gaussian(shape1, miu=0, sigma=0.5).astype(dtype1)
    input_x = random_gaussian(shape2, miu=0, sigma=0.5).astype(dtype2)
    expect = np.divide(1, np.add(1, np.square(input_x))) * head
    out_buf = np.full(shape1, np.nan, dtype1)
    return expect, (head, input_x), out_buf
