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
from tests.common.test_op.ascend.im2col import im2col_manual_schedule
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian


def im2col_benchmark(data, kernel, pad, stride):

    N, C1, H, W, C0 = data.shape
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel
    pad_t, pad_b, pad_l, pad_r = pad
    block_size = 16

    Ho = (H + pad_b + pad_t - kernel_h) // stride_h + 1
    Wo = (W + pad_r + pad_l - kernel_w) // stride_w + 1

    data_pad_shape = (N, C1, H + pad_t + pad_b, W + pad_l + pad_r, C0)
    data_pad = np.full(data_pad_shape, 0, dtype=data.dtype)
    data_pad[:, :, pad_t: pad_t + H, pad_l: pad_l + W, :] = data

    expect_shape = (N,
                    (Ho * Wo + block_size - 1) // block_size,
                    C1 * kernel_h * kernel_w,
                    block_size,
                    C0)
    expect = np.zeros(expect_shape, dtype=data.dtype)

    for n in range(N):
        for ho in range(Ho):
            for wo in range(Wo):
                for c1 in range(C1):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            expect[n, (ho*Wo+wo) // block_size, c1*kernel_h*kernel_w+kh*kernel_w+kw, (ho*Wo + wo) %
                                   block_size, :] = data_pad[n, c1, ho*stride_h + kh, wo*stride_w + kw, :]
    return expect


def im2col_run(shape, kernel, stride, pad, dtype, polyhedral=False, attrs=None):
    if polyhedral:
        raise Exception(
            "ERROR: no DSL with poly support for im2col, please select manual schedule version")
    else:
        mod = utils.op_build_test(im2col_manual_schedule, [shape],
                                  [dtype], kernel_name="im2col_manual_schedule",
                                  op_attrs=[kernel, stride, pad], attrs=attrs, polyhedral=polyhedral)
    expect, data, res = gen_data(dtype, kernel, pad, shape, stride)
    output = utils.mod_launch(mod, [data, res], expect=expect)
    atol, rtol = get_rtol_atol("im2col", dtype)
    return data, output, expect, compare_tensor(output, expect, atol=atol, rtol=rtol, equal_nan=True)


def gen_data(dtype, kernel, pad, shape, stride):
    data = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
    expect = im2col_benchmark(data, kernel, pad, stride).astype(dtype)
    res = np.full(expect.shape, np.nan, dtype)
    return expect, data, res
