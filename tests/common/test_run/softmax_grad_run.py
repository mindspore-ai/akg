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

"""softmax_grad_run"""

import numpy as np
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op import softmax_grad
from gen_random import random_gaussian

def softmax_grad_run(shape, dtype, axis, kernel_name, attrs=None):
    """run function for dsl function softmax_grad."""
    if attrs is None:
        attrs = {}

    input_shapes = [shape, shape]
    input_types = [dtype, dtype]
    op_attrs = [axis]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(softmax_grad.softmax_grad, input_shapes, input_types, op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            dy, expect, output, x = gen_data(axis, dtype, shape)
            return mod, expect, (x, dy, output)

        return mod

    dy, expect, output, x = gen_data(axis, dtype, shape)
    mod = utils.op_build_test(softmax_grad.softmax_grad, input_shapes, input_types, op_attrs,
                              kernel_name=kernel_name, attrs=attrs)
    output = utils.mod_launch(mod, (x, dy, output), expect=expect)
    return (x, dy), output, expect, compare_tensor(output, expect, rtol=5e-2, equal_nan=True)


def gen_data(axis, dtype, shape):
    """Generates input, output and expect data."""
    x = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    dy = random_gaussian(shape, miu=0, sigma=0.2).astype(dtype)
    x_sub = x - np.max(x, axis=axis, keepdims=True)
    x_sub_exp = np.exp(x_sub)
    y = x_sub_exp / np.sum(x_sub_exp, axis=axis, keepdims=True)
    y_grad = y * (1.0 - y)
    expect = dy * y_grad
    output = np.full(shape, -5.0, dtype)
    return dy, expect, output, x
