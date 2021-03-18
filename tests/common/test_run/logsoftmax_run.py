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
from tests.common.test_op import logsoftmax
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def logsoftmax_execute(shape, dtype, axis, kernel_name, attrs=None):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = logsoftmax_compile(shape, dtype, axis, kernel_name, attrs=None, tuning=t)
        if t:
            expect, input, output = method_name(axis, dtype, shape)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = logsoftmax_compile(shape, dtype, axis, kernel_name, attrs)
        expect, input, output = method_name(axis, dtype, shape)
        output = utils.mod_launch(mod, (input, output), expect=expect)
        rtol, atol = get_rtol_atol("logsoftmax", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def method_name(axis, dtype, shape):
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    sub = input - np.max(input, axis=axis, keepdims=True)
    e_x = np.exp(sub)
    logexpsum = np.log(np.sum(e_x, axis=axis, keepdims=True))
    expect = sub - logexpsum
    output = np.full(shape, np.nan, dtype)
    return expect, input, output


def logsoftmax_compile(shape, dtype, axis, kernel_name, attrs=None, tuning=False):
    ops_attrs = [axis]
    return utils.op_build_test(logsoftmax.logsoftmax, [shape], [dtype], ops_attrs, kernel_name, attrs=attrs, tuning=tuning)
