# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""
tanh grad run define
"""
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.math.ascend import TanhGrad
from tests.common.tensorio import compare_tensor
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian

def tanh_grad_execute(shape, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = tanh_grad_compile(shape, dtype, attrs, kernel_name=kernel_name, tuning=t)
        if t:
            expect, grad, input, output = gen_data(dtype, shape)
            return mod, expect, (input, grad, output)
        else:
            return mod
    else:
        mod = tanh_grad_compile(shape, dtype, attrs)
        expect, grad, input, output = gen_data(dtype, shape)
        output = utils.mod_launch(mod, (input, grad, output), expect=expect)
        rtol, atol = get_rtol_atol("tanh_grad", dtype)
        return (input, grad), output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, shape):
    support_list = {"float16": np.float16, "float32": np.float32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("tile_cce only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))
    # Generate data for testing the op
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    grad = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = (1.0 - np.power(input, 2)) * grad
    output = np.full(expect.shape, np.nan, dtype)
    return expect, grad, input, output


def tanh_grad_compile(shape, dtype, attrs, kernel_name='tanh_grad', tuning=False):
    return utils.op_build_test(TanhGrad, [shape, shape], [dtype, dtype], kernel_name=kernel_name, attrs=attrs, tuning=tuning)
