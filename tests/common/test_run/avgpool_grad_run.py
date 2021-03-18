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

"""
avgpool_grad run define
"""

import numpy as np
from akg.utils import kernel_exec as utils
from tests.common.test_run import avgpool_run
from tests.common.test_op import avgpool_grad
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from tests.common.tensorio import compare_tensor
from tests.common.gen_random import random_gaussian


def benchmark(dtype, x, y, dy, kernel, stride, pad):
    kh, kw = kernel
    sh, sw = stride
    N, C1, H, W, C0 = x.shape

    [ph_h, ph_t, pw_h, pw_t], _ = cal_pad_shapes_by_strategy(x.shape, kernel, stride, pad)

    padx = np.zeros((N, C1, H + ph_h + ph_t, W + pw_h + pw_t, C0), dtype=dtype)
    padx[:, :, ph_h:ph_h + H, pw_h:pw_h + W, :] = x

    dx = np.zeros_like(padx, dtype=dtype)
    _, _, yH, yW, _ = y.shape
    for i in range(yH):
        for j in range(yW):
            dx[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw, :] += (1.0 / (kh * kw)) * dy[:, :, i, j, :][:, :, None,
                                                                                                         None, :]

    return dx[:, :, ph_h:ph_h + H, pw_h:pw_h + W, :]


def avgpool_grad_run(shape, kernel, stride, pad, dtype, attrs):
    support_list = {"float16": np.float16}
    if not (dtype.lower() in support_list):
        raise RuntimeError("Auto-tensor only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        dy, x, y = get_input_data(dtype, kernel, pad, shape, stride, support_list)
        mod = utils.op_build_test(avgpool_grad.avgpool_grad, [x.shape, dy.shape], [dtype, dtype],
                                  op_attrs=[kernel, stride, pad], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, output = gen_data(dtype, dy, kernel, pad, shape, stride, x, y)
            return mod, expect, (x, dy, output)
        else:
            return mod
    else:
        dy, x, y = get_input_data(dtype, kernel, pad, shape, stride, support_list)
        mod = utils.op_build_test(avgpool_grad.avgpool_grad, [x.shape, dy.shape], [dtype, dtype],
                                  op_attrs=[kernel, stride, pad], kernel_name='avgpool_grad', attrs=attrs)
        expect, output = gen_data(dtype, dy, kernel, pad, shape, stride, x, y)
        output = utils.mod_launch(mod, (x, dy, output), expect=expect)

        return (x, y, dy), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, dy, kernel, pad, shape, stride, x, y):
    expect = benchmark(dtype, x, y, dy, kernel, stride, pad)
    output_shape = shape
    output = np.full(output_shape, 0.0, dtype)
    return expect, output


def get_input_data(dtype, kernel, pad, shape, stride, support_list):
    # Generate data for testing the op
    x = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    y = avgpool_run.benchmark(x, kernel, stride, pad)
    dy = random_gaussian(y.shape, miu=1, sigma=0.1).astype(support_list[dtype])
    dy = np.abs(dy)
    return dy, x, y
