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

from tests.common.tensorio import compare_tensor
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.nn.ascend import avgpool
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from tests.common.gen_random import random_gaussian


def benchmark(input, kernel, stride, pad):
    sh, sw = stride
    n, c1, h, w, c0 = input.shape
    KH, KW = kernel

    [ph_h, ph_t, pw_h, pw_t], [out_size_h, out_size_w] = cal_pad_shapes_by_strategy(input.shape, kernel, stride, pad)
    out_shape = (n, c1, out_size_h, out_size_w, c0)

    out = np.zeros(out_shape)

    inputpad = np.zeros((n, c1, h + ph_h + ph_t, w + pw_h + pw_t, c0))
    inputpad[:, :, ph_h:ph_h + h, pw_h:pw_h + w, :] = input

    for i in range(out_size_h):
        for j in range(out_size_w):
            out[:, :, i, j, :] = np.mean(inputpad[:, :, i * sh:i * sh + KH, j * sw:j * sw + KW, :], axis=(2, 3))
    return out


def avgpool_run(shape, kernel, stride, strategy, dtype, attrs):
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(avgpool, [shape], [dtype], op_attrs=[kernel, stride, strategy],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, output = gen_data(dtype, kernel, shape, strategy, stride)
            return mod, expect, (input, output)
        else:
            return mod
    else:
        mod = utils.op_build_test(avgpool, [shape], [dtype], op_attrs=[kernel, stride, strategy],
                                  kernel_name='avgpool', attrs=attrs)
        expect, input, output = gen_data(dtype, kernel, shape, strategy, stride)
        output = utils.mod_launch(mod, [input, output], expect=expect)
        return input, output, expect, compare_tensor(output, expect, rtol=5e-03, atol=5e-03, equal_nan=True)


def gen_data(dtype, kernel, shape, strategy, stride):
    support_list = {"float16": np.float16, "float32": np.float32}
    input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = benchmark(input, kernel, stride, strategy)
    out_shape = expect.shape
    output = np.full(out_shape, 0, dtype)
    return expect, input, output
