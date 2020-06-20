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
from test_op import softmax_ad
from gen_random import random_gaussian

def softmax_grad_expect(inputs, input_head):
    inputsSub = inputs - np.max(inputs, axis=-1, keepdims=True)
    inputsExp = np.exp(inputsSub)
    y = inputsExp / np.sum(inputsExp, axis=-1, keepdims=True)
    temp_shape = inputs.shape[:] + inputs.shape[-1:]
    grad = np.zeros(temp_shape)

    for k in range(temp_shape[-2]):
        for m in range(temp_shape[-1]):
            if k == m:
                grad[..., k, m] = y[..., k] * (1 - y[..., k])
            else:
                grad[..., k, m] = -y[..., k] * y[..., m]
            grad[..., k, m] = grad[..., k, m] * input_head[..., m]

    expect = np.sum(grad, axis=-1)

    return expect


def softmax_ad_run(shape, dtype, axis, kernel_name, optimized, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        if optimized:
            mod = utils.op_build_test(softmax_ad.softmax_ad_optimized, [shape, shape], [dtype, dtype],
                                      kernel_name=kernel_name, op_attrs=[axis], attrs=attrs, tuning=t)
        else:
            mod = utils.op_build_test(softmax_ad.softmax_ad, [shape, shape], [dtype, dtype], kernel_name=kernel_name,
                                      op_attrs=[axis], attrs=attrs, tuning=t)
        if t:
            expect, input_head, inputs, output = gen_data(dtype, shape)
            return mod, expect, (input_head, inputs, output)
        else:
            return mod
    else:
        expect, input_head, inputs, output = gen_data(dtype, shape)
        if optimized:
            mod = utils.op_build_test(softmax_ad.softmax_ad_optimized, [shape, shape], [dtype, dtype],
                                      kernel_name="softmax_ad_optimized", op_attrs=[axis], attrs=attrs)
        else:
            mod = utils.op_build_test(softmax_ad.softmax_ad, [shape, shape], [dtype, dtype], kernel_name="softmax_ad",
                                      op_attrs=[axis], attrs=attrs)
        print(mod.imported_modules[0].get_source())

        output = utils.mod_launch(mod, [input_head, inputs, output], expect=expect)
        return [input_head, inputs], output, expect, np.allclose(output, expect, rtol=5e-03, atol=0.1, equal_nan=True)


def gen_data(dtype, shape):
    inputs = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    input_head = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    input_head = np.abs(input_head)
    expect = softmax_grad_expect(inputs, input_head)
    e_shape = expect.shape
    output = np.full(e_shape, 1.0, dtype)
    return expect, input_head, inputs, output
