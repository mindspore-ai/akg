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
from akg.utils import kernel_exec as utils
from akg.ops.math import tanh
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def tanh_execute(shape, dtype, attrs=None):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = tanh_compile(shape, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = tanh_compile(shape, dtype, attrs)
        expect, input, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)  # unified launch
        rtol, atol = get_rtol_atol("tanh", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape):
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    expect = np.tanh(input)
    output = np.full(expect.shape, np.nan, dtype)
    return expect, input, output


def tanh_compile(shape, dtype, attrs, kernel_name='tanh', tuning=False):
    return utils.op_build_test(tanh.tanh, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
