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
from tests.common.test_op import logsoftmax_grad
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def logsoftmax_grad_execute(shape, dtype, axis, kernel_name, attrs=None):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = logsoftmax_grad_compile(shape, dtype, axis, kernel_name, attrs=None, tuning=t)
        if t:
            expect, grad, input, output = gen_data(axis, dtype, shape)
            return mod, expect, (input, grad, output)
        else:
            return mod
    else:
        mod = logsoftmax_grad_compile(shape, dtype, axis, kernel_name, attrs)
        expect, grad, input, output = gen_data(axis, dtype, shape)
        output = utils.mod_launch(mod, (input, grad, output), expect=expect)
        rtol, atol = get_rtol_atol("logsoftmax_grad", dtype)
        return (input, grad), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)

def akg_sum(input, axis, keepdims):
    if input.dtype == "float16":
        return np.sum(input.astype("float32"), axis=axis, keepdims=keepdims).astype("float16")
    else:
        return np.sum(input, axis=axis, keepdims=keepdims)

def gen_data(axis, dtype, shape):
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    grad = random_gaussian(shape, miu=0, sigma=0.2).astype(dtype)
    expect = grad - np.exp(input) * akg_sum(grad, axis=axis, keepdims=True)
    output = np.full(shape, np.nan, dtype)
    return expect, grad, input, output


def logsoftmax_grad_compile(shape, dtype, axis, kernel_name, attrs=None, tuning=False):
    return utils.op_build_test(logsoftmax_grad.logsoftmax_grad, [shape, shape], [dtype, dtype], op_attrs=[axis],
                               kernel_name=kernel_name, attrs=attrs, tuning=tuning)
