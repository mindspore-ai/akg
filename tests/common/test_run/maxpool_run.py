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

from tests.common.tensorio import compare_tensor
import numpy as np
from akg.utils import kernel_exec as utils
from akg.ops.nn import maxpool
from akg.utils.dsl_create import cal_pad_shapes_by_strategy
from akg import tvm
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def benchmark(input, kernel, stride, pad):
    sh, sw = stride
    N, C1, H, W, C0 = input.shape
    KH, KW = kernel

    [ph_h, ph_t, pw_h, pw_t], [out_size_h, out_size_w] = cal_pad_shapes_by_strategy(input.shape, kernel, stride, pad)

    out_shape = (N, C1, out_size_h, out_size_w, C0)
    out = np.zeros(out_shape)

    inputpad = np.full((N, C1, H + ph_h + ph_t, W + pw_h + pw_t, C0), np.finfo(input.dtype).min, dtype=input.dtype)
    inputpad[:, :, ph_h:ph_h + H, pw_h:pw_h + W, :] = input

    for i in range(out_size_h):
        for j in range(out_size_w):
            out[:, :, i, j, :] = np.max(inputpad[:, :, i * sh:i * sh + KH, j * sw:j * sw + KW, :], axis=(2, 3))
    return out


def maxpool_run(shape, kernel, stride, pad, hybrid, dtype, attrs=None, polyhedral=True):
    if attrs.get("dynamic"):
        var_shape = []
        for i in range(len(shape)):
            if i == len(shape) - 1:
                var_shape.append(shape[i])
            else:
                var_shape.append(tvm.var("I" + str(i)))
        build_shape = var_shape
    else:
        build_shape = shape
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(maxpool.maxpool, [build_shape], [dtype], op_attrs=[kernel, stride, pad],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            expect, input, out_shape, res = gen_data(dtype, kernel, pad, shape, stride)
            return mod, expect,  {"args": (input, res), 'outputs': (-1, ), 'tuning': False}
        else:
            return mod
    else:
        if polyhedral:
            if hybrid:
                mod = utils.op_build_test(maxpool.maxpool, [build_shape], [dtype], op_attrs=[kernel, stride, pad],
                                          kernel_name='maxpool', attrs=attrs)
            else:
                mod = utils.op_build_test(maxpool.old_maxpool, [build_shape], [dtype], op_attrs=[kernel, stride, pad],
                                          kernel_name='maxpool_old', attrs=attrs)
        else:
            mod = maxpool.maxpool_manual_schedule(build_shape, kernel, stride, pad, dtype,attrs=attrs, polyhedral=polyhedral)
        expect, input, out_shape, res = gen_data(dtype, kernel, pad, shape, stride)
        output = utils.mod_launch(mod, [input, res], expect=expect)
        rtol, atol = get_rtol_atol("maxpool", dtype)
        return input, output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)


def gen_data(dtype, kernel, pad, shape, stride):
    support_list = {"float16": np.float16, "float32": np.float32}
    import time
    seed_tmp = int(time.time())
    input = random_gaussian(shape, miu=1, sigma=0.1, seed=seed_tmp).astype(support_list[dtype])
    input = -input
    expect = benchmark(input, kernel, stride, pad)
    out_shape = expect.shape
    #  print("INFO Test data seed: ", seed_tmp)
    res = np.full(out_shape, np.nan, dtype)
    return expect, input, out_shape, res
